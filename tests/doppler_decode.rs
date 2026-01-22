use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;

fn apply_ou_doppler_and_awgn(
    x: &[Complex32],
    fs_hz: f64,
    cfo_hz: f64,
    doppler_std_hz: f64,
    doppler_tau_s: f64,
    noise_std: f32,
    seed: u64,
) -> Vec<Complex32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n01 = Normal::<f32>::new(0.0, 1.0).unwrap();

    // OU process
    let a = if doppler_tau_s > 0.0 {
        (-1.0 / (fs_hz * doppler_tau_s)).exp()
    } else {
        0.0
    };
    let b = doppler_std_hz * (1.0 - a * a).max(0.0).sqrt();
    let mut f_hz = 0.0f64;
    let mut phi = 0.0f32;

    let mut y = Vec::with_capacity(x.len());
    for &s in x {
        let w = n01.sample(&mut rng) as f64;
        f_hz = a * f_hz + b * w;
        let inst_f_hz = cfo_hz + f_hz;

        let mut v = s;
        if inst_f_hz != 0.0 {
            v *= Complex32::from_polar(1.0, phi);
            let dphi = (2.0 * std::f64::consts::PI * inst_f_hz / fs_hz) as f32;
            phi += dphi;
            if phi.abs() > 1000.0 {
                phi = phi.rem_euclid(2.0 * std::f32::consts::PI);
            }
        }

        if noise_std > 0.0 {
            let nr = noise_std * n01.sample(&mut rng);
            let ni = noise_std * n01.sample(&mut rng);
            v += Complex32::new(nr, ni);
        }
        y.push(v);
    }
    y
}

#[test]
fn test_decode_with_doppler_ou() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    // Make t_tx land at n0=4 samples into the IV epoch (like the CLI example).
    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs; // within [0,1ms)
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    // Build a raw window that contains the whole frame starting at `pre`.
    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let pad_post = 512usize; // extra samples so interpolation never runs off the end
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);

    // Apply channel impairments.
    let raw = apply_ou_doppler_and_awgn(
        &raw, fs, 0.0, 1.0, // doppler_std_hz
        1.0, // doppler_tau_s
        2.0, // noise_std
        12345,
    );

    // Acquisition window must include all pilots for hybrid verification.
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let acq_win = &raw[..win_need];
    let acq = modem
        .acquire_fft_raw_window(acq_win, ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    assert_eq!(acq.ti_hat, ti_tx);
    assert_eq!(acq.n0, n0_true);

    // Demod using acquisition outputs.
    let frame_start_sample = base; // epoch start; add n0 via offsets
    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        frame_start_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_doppler_ou_no_noise() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);

    let raw = apply_ou_doppler_and_awgn(&raw, fs, 0.0, 1.0, 1.0, 0.0, 12345);

    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    let (payload, meta) =
        modem.demod_decode_raw(&raw, ti_tx, base, &acq.finger_offsets, acq.cfo_hat_hz, 20)?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_awgn_only() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);

    let raw = apply_ou_doppler_and_awgn(&raw, fs, 0.0, 0.0, 1.0, 2.0, 12345);

    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    let (payload, meta) =
        modem.demod_decode_raw(&raw, ti_tx, base, &acq.finger_offsets, acq.cfo_hat_hz, 20)?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_constant_cfo_no_noise() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);

    let cfo_hz = 10.0;
    let raw = apply_ou_doppler_and_awgn(&raw, fs, cfo_hz, 0.0, 1.0, 0.0, 12345);

    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    assert!((acq.cfo_hat_hz - cfo_hz).abs() < 1.0, "acq={acq:?}");

    let (payload, meta) =
        modem.demod_decode_raw(&raw, ti_tx, base, &acq.finger_offsets, acq.cfo_hat_hz, 20)?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_constant_cfo_without_derotation() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);

    let cfo_hz = 2.0;
    let raw = apply_ou_doppler_and_awgn(&raw, fs, cfo_hz, 0.0, 1.0, 0.0, 12345);

    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    // Intentionally do NOT derotate with the coarse CFO estimate; force the symbol-rate loop
    // to handle it.
    let (payload, meta) =
        modem.demod_decode_raw(&raw, ti_tx, base, &acq.finger_offsets, 0.0, 50)?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}
