use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;
use sc_bltc::sim::channel::{
    apply_channel_sample, normalize_taps, resample_linear, ChannelState, Gauss, MultipathFir, Rng64,
};

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

fn decode_sro_case(sro_ppm: f64) -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p, [0u8; 32])?;
    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let t_tx = 4000.0 + 4.0 / fs;
    let frame = modem.build_frame_samples(b"sro", 1, 1, Some(t_tx))?;
    let ti_min = frame.ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((frame.ti_tx - ti_min) as usize) * iv_samples;
    let tx = resample_linear(&frame.samples, 1.0 + sro_ppm * 1e-6)?;
    let pre = base + 4;
    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;
    let mut raw = vec![Complex32::new(0.0, 0.0); win_need.max(pre + tx.len() + 512)];
    raw[pre..pre + tx.len()].copy_from_slice(&tx);

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed at sro_ppm={sro_ppm}"))?;
    assert_eq!(acq.ti_hat, frame.ti_tx, "sro_ppm={sro_ppm}, acq={acq:?}");
    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        frame.ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;
    assert!(meta.crc_ok, "sro_ppm={sro_ppm}, meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"sro".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_doppler_ou() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    // Make t_tx land at n0=4 samples into the IV epoch (like the CLI example).
    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs; // within [0,1ms)
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    // Build a raw window that contains the whole frame starting at `pre`.
    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize; // extra samples so interpolation never runs off the end
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    // Apply channel impairments.
    let raw = apply_ou_doppler_and_awgn(
        &raw, fs, 0.0, 1.0, // doppler_std_hz
        1.0, // doppler_tau_s
        2.0, // noise_std
        12345,
    );

    let acq_win = &raw[..win_need];
    let acq = modem
        .acquire_fft_raw_window(acq_win, ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    assert_eq!(acq.ti_hat, ti_tx);
    assert!(
        (acq.arrival_sample as isize - (base + n0_true) as isize).abs() <= 3,
        "acq={acq:?}"
    );

    // Demod using acquisition outputs.
    let frame_start_sample = acq.arrival_sample;
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
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let raw = apply_ou_doppler_and_awgn(&raw, fs, 0.0, 1.0, 1.0, 0.0, 12345);

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_awgn_only() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let raw = apply_ou_doppler_and_awgn(&raw, fs, 0.0, 0.0, 1.0, 2.0, 12345);

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_constant_cfo_no_noise() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let cfo_hz = 10.0;
    let raw = apply_ou_doppler_and_awgn(&raw, fs, cfo_hz, 0.0, 1.0, 0.0, 12345);

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    assert!((acq.cfo_hat_hz - cfo_hz).abs() < 1.0, "acq={acq:?}");

    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_constant_cfo_without_derotation() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let cfo_hz = 2.0;
    let raw = apply_ou_doppler_and_awgn(&raw, fs, cfo_hz, 0.0, 1.0, 0.0, 12345);

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    // Intentionally do NOT derotate with the coarse CFO estimate; force the symbol-rate loop
    // to handle it.
    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        0.0,
        50,
    )?;

    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"test".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_sampling_rate_offset() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"sro", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;
    let sro_ppm = 100.0;
    let sample_rate_ratio = 1.0 + sro_ppm * 1e-6;
    let tx_samples = resample_linear(&tx_frame.samples, sample_rate_ratio)?;
    assert!(tx_samples.len() > tx_frame.samples.len());

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;
    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 64;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_samples.len() + 512];
    raw[pre..pre + tx_samples.len()].copy_from_slice(&tx_samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed at sro_ppm={sro_ppm}"))?;
    assert_eq!(acq.ti_hat, ti_tx);
    assert!(
        (acq.arrival_sample as isize - (base + n0_true) as isize).abs() <= 20,
        "acq={acq:?}"
    );

    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;
    assert!(meta.crc_ok, "decode failed: meta={meta:?}, acq={acq:?}");
    assert_eq!(payload.as_deref(), Some(b"sro".as_slice()));
    Ok(())
}

#[test]
fn test_decode_with_extreme_sampling_rate_offsets() -> anyhow::Result<()> {
    for sro_ppm in [-500.0, 350.0, 500.0] {
        decode_sro_case(sro_ppm)?;
    }
    Ok(())
}

#[test]
fn test_decode_with_joint_tracking_sro_and_multipath() -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p, [0u8; 32])?;
    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;
    let t_tx = 4000.0f64 + 4.0 / fs;
    let frame = modem.build_frame_samples(b"joint", 1, 1, Some(t_tx))?;
    let ti_tx = frame.ti_tx;
    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + 4;
    let sro_ppm = 300.0f64;
    let tx = resample_linear(&frame.samples, 1.0 + sro_ppm * 1e-6)?;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need =
        n_ti * iv_samples + 2 * rake_search_half + p.frame_max_samples_with_jitter() + 128;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx.len() + 512];
    raw[pre..pre + tx.len()].copy_from_slice(&tx);
    raw.resize(raw.len().max(win_need), Complex32::new(0.0, 0.0));

    let mut taps = vec![Complex32::new(0.0, 0.0); 101];
    taps[0] = Complex32::new(1.0, 0.0);
    taps[40] = Complex32::new(10.0f32.powf(-3.0 / 20.0), 0.0);
    taps[100] = Complex32::new(10.0f32.powf(-8.0 / 20.0), 0.0);
    let mut channel = ChannelState::new(Some(MultipathFir::new(normalize_taps(taps))?), None);
    let mut rng = Rng64::new(1);
    let mut gauss = Gauss::new();
    let raw: Vec<_> = raw
        .into_iter()
        .map(|x| {
            apply_channel_sample(
                x,
                1.0,
                0.0,
                fs * (1.0 + sro_ppm * 1e-6),
                0.0,
                &mut channel,
                &mut rng,
                &mut gauss,
            )
        })
        .collect();

    let acq = modem
        .acquire_fft_raw_window(&raw[..win_need], ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;
    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        acq.arrival_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;
    assert!(
        meta.crc_ok,
        "joint tracking failed: meta={meta:?}, acq={acq:?}"
    );
    assert_eq!(payload.as_deref(), Some(b"joint".as_slice()));
    Ok(())
}
