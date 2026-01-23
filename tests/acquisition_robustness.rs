use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;

#[test]
fn test_acquire_on_noise() {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key).unwrap();

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;

    let n_ti = 1;
    let window_len = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;

    let mut rng = StdRng::seed_from_u64(0);
    let n01 = Normal::<f32>::new(0.0, 1.0).unwrap();
    let noise: Vec<Complex32> = (0..window_len)
        .map(|_| {
            let re: f32 = n01.sample(&mut rng);
            let im: f32 = n01.sample(&mut rng);
            Complex32::new(re, im)
        })
        .collect();

    let result = modem
        .acquire_fft_matched_window(&noise, 1000, n_ti, 1e-3, 3)
        .unwrap();
    assert!(result.is_none(), "false alarm: {result:?}");
}

#[test]
fn test_acquire_raw_noise_plus_cfo() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p.clone(), key)?;

    let noise_std: f32 = 2.0;
    let cfo_hz: f64 = 50.0;

    let t_tx = 4000.0f64 + 0.00037;
    let frame = modem.build_frame_samples(b"t", 1, 1, Some(t_tx))?;
    let ti_tx = frame.ti_tx;

    let fs = p.fs_hz as f64;
    let iv_samples = (fs * p.iv_res_s).round() as usize;
    let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
    let l_sym = p.chip_samples();
    let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
    let last_pilot_end = (ell_last_pilot + 1) * l_sym;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
    let n0 = (frac * fs).round() as usize % iv_samples;
    let pre = base + n0;

    let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;
    let mut raw = vec![Complex32::new(0.0, 0.0); win_need];

    let mut rng = StdRng::seed_from_u64(1);
    let n01 = Normal::<f32>::new(0.0, noise_std)?;
    for s in &mut raw {
        *s = Complex32::new(n01.sample(&mut rng), n01.sample(&mut rng));
    }

    let sig_need = (last_pilot_end + 32).min(frame.samples.len());
    raw[pre..pre + sig_need].copy_from_slice(&frame.samples[..sig_need]);

    // Apply a global CFO rotation to the entire window (signal + noise).
    let dphi = (2.0 * std::f64::consts::PI * cfo_hz / fs) as f32;
    let mut phi = 0.0f32;
    for s in &mut raw {
        *s *= Complex32::from_polar(1.0, phi);
        phi += dphi;
    }

    let acq = modem
        .acquire_fft_raw_window(&raw, ti_min, n_ti, 1e-9, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed"))?;

    assert_eq!(acq.ti_hat, ti_tx, "acq={acq:?}, ti_tx={ti_tx}");
    assert_eq!(acq.n0, n0, "acq={acq:?}, n0_true={n0}");
    assert!(
        (acq.cfo_hat_hz - cfo_hz).abs() <= 5.0,
        "acq={acq:?}, cfo_true={cfo_hz}"
    );
    Ok(())
}

// #[test]
// fn test_key_mismatch() {
//     let p = Params::default();
//     let key_a = [0xAAu8; 32];
//     let key_b = [0xBBu8; 32];

//     let modem_a = ScBltcModem::new(p.clone(), key_a).unwrap();
//     let modem_b = ScBltcModem::new(p.clone(), key_b).unwrap();

//     let t_tx = 1000.0f64;
//     let tx_frame = modem_a
//         .build_frame_samples(b"test", 1, 1, Some(t_tx))
//         .unwrap();

//     let ti_tx = tx_frame.ti_tx;
//     let fs = p.fs_hz as f64;
//     let iv_samples = (fs * p.iv_res_s).round() as usize;
//     let rake_search_half = (fs * p.rake_search_half_s).round() as usize;
//     let frac = t_tx - (ti_tx as f64) * p.iv_res_s;
//     let n0 = (frac * fs).round() as usize % iv_samples;

//     let ti_min = ti_tx.saturating_sub(2);
//     let n_ti = 5usize;
//     let base = ((ti_tx - ti_min) as usize) * iv_samples;
//     let pre = base + n0;

//     let l_sym = p.chip_samples();
//     let ell_last_pilot = 2 + 5 * (p.n_pilot - 1);
//     let last_pilot_end = (ell_last_pilot + 1) * l_sym;
//     let win_need = n_ti * iv_samples + iv_samples + rake_search_half + last_pilot_end + 32;

//     let mut rx_raw = vec![Complex32::new(0.0, 0.0); win_need];
//     let sig_need = (last_pilot_end + 32).min(tx_frame.samples.len());
//     rx_raw[pre..pre + sig_need].copy_from_slice(&tx_frame.samples[..sig_need]);
//     let mut rng = StdRng::seed_from_u64(2);
//     let rx_noisy: Vec<Complex32> = rx_raw
//         .iter()
//         .map(|s| {
//             let n_re = rng.random_range(-0.01..0.01);
//             let n_im = rng.random_range(-0.01..0.01);
//             s + Complex32::new(n_re, n_im)
//         })
//         .collect();

//     let result = modem_b
//         .acquire_fft_raw_window(&rx_noisy, ti_min, n_ti, 1e-3, 3)
//         .unwrap();
//     assert!(result.is_none(), "acquired with wrong key: {result:?}");
// }

#[test]
fn test_crypto_different_keys() {
    use sc_bltc::crypto::gen_code_aes_ctr;
    let key_a = [0xAAu8; 32];
    let key_b = [0xBBu8; 32];
    let ti = 1000;
    let len = 1024;
    let domain = 0x12345678;

    let c_a = gen_code_aes_ctr(&key_a, ti, len, domain);
    let c_b = gen_code_aes_ctr(&key_b, ti, len, domain);

    assert_ne!(c_a, c_b, "Different keys must produce different codes");

    let mut dot_prod = 0;
    for i in 0..len {
        dot_prod += (c_a[i] as i32) * (c_b[i] as i32);
    }

    if dot_prod.abs() > 200 {
        panic!(
            "Surprisingly high correlation between random codes: {}",
            dot_prod
        );
    }
}
