use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;

fn add_awgn(x: &[Complex32], noise_std: f32, seed: u64) -> Vec<Complex32> {
    if noise_std <= 0.0 {
        return x.to_vec();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n01 = Normal::<f32>::new(0.0, noise_std).unwrap();
    x.iter()
        .map(|&s| Complex32::new(s.re + n01.sample(&mut rng), s.im + n01.sample(&mut rng)))
        .collect()
}

/// Manual performance sanity-check: run with:
/// `cargo test -q --test noise_margin -- --ignored --nocapture`
#[test]
#[ignore]
fn decode_awgn_noise_std_6_single_seed() -> anyhow::Result<()> {
    let p = Params::default();
    let key = [0u8; 32];
    let modem = ScBltcModem::new(p, key)?;

    let fs = p.fs_hz() as f64;
    let iv_samples = (fs * p.iv_res_s()).round() as usize;

    // Place frame at a deterministic offset within the IV epoch.
    let n0_true = 4usize;
    let t_tx = 4000.0f64 + (n0_true as f64) / fs;
    let tx_frame = modem.build_frame_samples(b"test", 1, 1, Some(t_tx))?;
    let ti_tx = tx_frame.ti_tx;

    let ti_min = ti_tx.saturating_sub(2);
    let n_ti = 5usize;
    let base = ((ti_tx - ti_min) as usize) * iv_samples;
    let pre = base + n0_true;

    let rake_search_half = (fs * p.rake_search_half_s()).round() as usize;
    let win_need = n_ti * iv_samples + rake_search_half + p.frame_max_samples_with_jitter() + 64;

    let pad_post = 512usize;
    let mut raw = vec![Complex32::new(0.0, 0.0); pre + tx_frame.samples.len() + pad_post];
    raw[pre..pre + tx_frame.samples.len()].copy_from_slice(&tx_frame.samples);
    if raw.len() < win_need {
        raw.resize(win_need, Complex32::new(0.0, 0.0));
    }

    let noise_std = 6.0f32;
    let raw = add_awgn(&raw, noise_std, 12345);

    let acq_win = &raw[..win_need];
    let acq = modem
        .acquire_fft_raw_window(acq_win, ti_min, n_ti, 3)?
        .ok_or_else(|| anyhow::anyhow!("acq_failed at noise_std={noise_std}"))?;

    eprintln!("acq={acq:?}");

    let frame_start_sample = base; // epoch start; add n0 via offsets
    let (payload, meta) = modem.demod_decode_raw(
        &raw,
        ti_tx,
        frame_start_sample,
        &acq.finger_offsets,
        acq.cfo_hat_hz,
        20,
    )?;
    eprintln!("meta={meta:?}, payload={:?}", payload.as_deref());
    Ok(())
}
