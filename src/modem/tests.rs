use super::*;
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[test]
fn end_to_end_ideal_frame() -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p, [0u8; 32])?;

    let frame = modem.build_frame_samples(b"hello", 1, 1, Some(1000.0))?;

    let mut rx = frame.samples.clone();
    rx.extend(std::iter::repeat(Complex32::new(0.0, 0.0)).take(2 * modem.rrc.delay() + 16));

    let (pl, meta) = modem.demod_decode_raw(&rx, frame.ti_tx, 0, &[0], 0.0, 8)?;
    assert!(meta.crc_ok, "meta={meta:?}");
    assert_eq!(pl.unwrap(), b"hello");
    Ok(())
}

#[test]
fn end_to_end_with_known_cfo() -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p.clone(), [0u8; 32])?;

    let frame = modem.build_frame_samples(b"cfo", 1, 1, Some(2000.0))?;

    let cfo_hz = 50.0f64;
    let fs = p.fs_hz as f32;
    let mut rx: Vec<Complex32> = frame
        .samples
        .iter()
        .enumerate()
        .map(|(n, &x)| {
            let ph = 2.0 * std::f32::consts::PI * (cfo_hz as f32) * (n as f32) / fs;
            x * Complex32::from_polar(1.0, ph)
        })
        .collect();
    rx.extend(std::iter::repeat(Complex32::new(0.0, 0.0)).take(2 * modem.rrc.delay() + 16));

    let (pl, meta) = modem.demod_decode_raw(&rx, frame.ti_tx, 0, &[0], cfo_hz, 8)?;
    assert!(meta.crc_ok, "meta={meta:?}");
    assert_eq!(pl.unwrap(), b"cfo");
    Ok(())
}

#[test]
fn acquisition_fft_then_decode() -> anyhow::Result<()> {
    // Spec §4.B.1 (epoch-internal offset).
    let p = Params::default();
    let modem = ScBltcModem::new(p.clone(), [0u8; 32])?;

    let t_tx = 4000.0f64 + 0.00037;
    let frame = modem.build_frame_samples(b"acq", 1, 1, Some(t_tx))?;

    let ti_tx = frame.ti_tx;
    let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as usize;
    let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
    let n0 = (frac * (p.fs_hz as f64)).round() as usize % iv_samples;

    let ti_min = ti_tx.saturating_sub(2);
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0;

    let cfo_hz = 40.0f64;
    let fs = p.fs_hz as f32;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre];
    raw.extend(frame.samples.iter().enumerate().map(|(n, &x)| {
        let ph = 2.0 * std::f32::consts::PI * (cfo_hz as f32) * (n as f32) / fs;
        x * Complex32::from_polar(1.0, ph)
    }));

    let y = modem.rrc.filter_same(&raw);
    let n_ti = 5usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = ((p.fs_hz as f64) * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let y_win = &y[..win_need];

    let acq = modem
        .acquire_fft_matched_window(y_win, ti_min, n_ti, 1e-3, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;
    assert_eq!(acq.ti_hat, ti_tx);
    assert_eq!(acq.n0, n0);

    let (pl, meta) = modem.demod_decode_raw(
        &raw,
        acq.ti_hat,
        base,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        16,
    )?;
    assert!(meta.crc_ok, "meta={meta:?}, acq={acq:?}");
    assert_eq!(pl.unwrap(), b"acq");
    Ok(())
}

#[test]
fn acquisition_fft_large_cfo_then_decode() -> anyhow::Result<()> {
    // Spec §4.B.1 (CFO search band).
    let p = Params::default();
    let modem = ScBltcModem::new(p.clone(), [0u8; 32])?;

    let t_tx = 5000.0f64 + 0.00037;
    let frame = modem.build_frame_samples(b"acq2", 1, 1, Some(t_tx))?;

    let ti_tx = frame.ti_tx;
    let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as usize;
    let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
    let n0 = (frac * (p.fs_hz as f64)).round() as usize % iv_samples;

    let ti_min = ti_tx.saturating_sub(2);
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0;

    let cfo_hz = 5001.0f64;
    let fs = p.fs_hz as f32;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre];
    raw.extend(frame.samples.iter().enumerate().map(|(n, &x)| {
        let ph = 2.0 * std::f32::consts::PI * (cfo_hz as f32) * (n as f32) / fs;
        x * Complex32::from_polar(1.0, ph)
    }));

    let n_ti = 5usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = ((p.fs_hz as f64) * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let x_win = &raw[..win_need];

    let acq = modem
        .acquire_fft_raw_window(x_win, ti_min, n_ti, 1e-3, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;
    assert_eq!(acq.ti_hat, ti_tx);
    assert!((acq.n0 as isize - n0 as isize).abs() <= 3, "acq={acq:?}");
    assert!((acq.cfo_hat_hz - cfo_hz).abs() <= 50.0, "acq={acq:?}");

    let (pl, meta) = modem.demod_decode_raw(
        &raw,
        acq.ti_hat,
        base,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        16,
    )?;
    assert!(meta.crc_ok, "meta={meta:?}, acq={acq:?}");
    assert_eq!(pl.unwrap(), b"acq2");
    Ok(())
}

#[test]
fn end_to_end_with_random_doppler() -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p.clone(), [0u8; 32])?;

    let path1_amp: f32 = 1.0;
    let path1_doppler_hz: f64 = 0.0;
    let path1_phase0: f64 = 0.0;

    let path2_amp: f32 = 0.95;
    let path2_doppler_hz: f64 = 1.5;
    let path2_phase0: f64 = std::f64::consts::PI / 4.0;
    let path2_delay_samples: usize = 6;

    let noise_std: f32 = 0.5;

    let fs = p.fs_hz as f64;
    let t_tx = 6000.0f64 + 18.0 / fs;
    let payload = b"doppler";
    let frame = modem.build_frame_samples(payload, 1, 1, Some(t_tx))?;

    let ti_tx = frame.ti_tx;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
    let n0 = (frac * fs).round() as usize % iv_samples;

    let ti_min = ti_tx.saturating_sub(2);
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0;

    let mut signal = vec![Complex32::new(0.0, 0.0); pre];
    signal.extend_from_slice(&frame.samples);
    signal.extend(std::iter::repeat(Complex32::new(0.0, 0.0)).take(2 * modem.rrc.delay() + 256));

    let n_samples = signal.len();
    let mut raw = vec![Complex32::new(0.0, 0.0); n_samples];
    let two_pi = 2.0 * std::f64::consts::PI;

    for n in 0..n_samples {
        let t = n as f64 / fs;
        let rot1 = Complex32::from_polar(
            path1_amp,
            (two_pi * path1_doppler_hz * t + path1_phase0) as f32,
        );
        let x1 = signal[n] * rot1;

        let x2 = if n >= path2_delay_samples {
            let rot2 = Complex32::from_polar(
                path2_amp,
                (two_pi * path2_doppler_hz * t + path2_phase0) as f32,
            );
            signal[n - path2_delay_samples] * rot2
        } else {
            Complex32::new(0.0, 0.0)
        };
        raw[n] = x1 + x2;
    }

    let mut rng = StdRng::seed_from_u64(1);
    let n = Normal::<f32>::new(0.0, noise_std)?;
    for x in &mut raw {
        *x += Complex32::new(n.sample(&mut rng), n.sample(&mut rng));
    }

    let n_ti = 5usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = ((p.fs_hz as f64) * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let x_win = &raw[..win_need];

    let acq = modem
        .acquire_fft_raw_window(x_win, ti_min, n_ti, 1e-3, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;
    assert_eq!(acq.ti_hat, ti_tx);
    assert!((acq.n0 as isize - n0 as isize).abs() <= 3, "acq={acq:?}");

    let (pl, meta) = modem.demod_decode_raw(
        &raw,
        acq.ti_hat,
        base,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        32,
    )?;
    assert!(meta.crc_ok, "meta={meta:?}, acq={acq:?}");
    assert_eq!(pl.unwrap(), payload);
    Ok(())
}

#[test]
fn acquisition_finds_multipath_beyond_one_iv() -> anyhow::Result<()> {
    let p = Params::default();
    let modem = ScBltcModem::new(p.clone(), [0u8; 32])?;

    let path1_amp: f32 = 1.0;
    let path2_amp: f32 = 0.9;
    let noise_std: f32 = 0.15;

    let fs = p.fs_hz as f64;
    let t_tx = 7000.0f64 + 18.0 / fs;
    let payload = b"mpath";
    let frame = modem.build_frame_samples(payload, 1, 1, Some(t_tx))?;

    let ti_tx = frame.ti_tx;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
    let n0 = (frac * fs).round() as usize % iv_samples;

    let ti_min = ti_tx.saturating_sub(2);
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0;

    // Delay the second path beyond one IV but still within the default RAKE search window.
    let path2_delay_samples: usize = iv_samples + 50;

    let mut signal = vec![Complex32::new(0.0, 0.0); pre];
    signal.extend_from_slice(&frame.samples);
    signal.extend(std::iter::repeat(Complex32::new(0.0, 0.0)).take(2 * modem.rrc.delay() + 512));

    let n_samples = signal.len();
    let mut raw = vec![Complex32::new(0.0, 0.0); n_samples];
    for n in 0..n_samples {
        let x1 = signal[n] * path1_amp;
        let x2 = if n >= path2_delay_samples {
            signal[n - path2_delay_samples] * path2_amp
        } else {
            Complex32::new(0.0, 0.0)
        };
        raw[n] = x1 + x2;
    }

    let mut rng = StdRng::seed_from_u64(2);
    let n = Normal::<f32>::new(0.0, noise_std)?;
    for x in &mut raw {
        *x += Complex32::new(n.sample(&mut rng), n.sample(&mut rng));
    }

    let n_ti = 5usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let x_win = &raw[..win_need];

    let acq = modem
        .acquire_fft_raw_window(x_win, ti_min, n_ti, 1e-3, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;
    assert_eq!(acq.ti_hat, ti_tx);
    assert!((acq.n0 as isize - n0 as isize).abs() <= 3, "acq={acq:?}");

    let want = n0 + path2_delay_samples;
    assert!(
        acq.finger_offsets
            .iter()
            .any(|&off| (off as isize - want as isize).abs() <= 2),
        "expected a finger near {}, got {:?} (n0={})",
        want,
        acq.finger_offsets,
        n0
    );

    let (pl, meta) = modem.demod_decode_raw(
        &raw,
        acq.ti_hat,
        base,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        32,
    )?;
    assert!(meta.crc_ok, "meta={meta:?}, acq={acq:?}");
    assert_eq!(pl.unwrap(), payload);
    Ok(())
}
