use anyhow::Context;
use clap::Parser;
use num_complex::Complex32;
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;
use sc_bltc::ring::Ring;
use std::collections::HashMap;
use std::io::{self, Read};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const MAGIC: &[u8; 8] = b"SCBLTC01";

#[derive(Parser, Debug)]
#[command(about = "SC-BLTC receiver over TCP (no system audio stack)")]
struct Args {
    /// Bind address, e.g. 0.0.0.0:5555 or 127.0.0.1:5555
    #[arg(long, default_value = "127.0.0.1:5555")]
    bind: String,

    #[arg(
        long,
        default_value = "0000000000000000000000000000000000000000000000000000000000000000"
    )]
    key_hex: String,

    /// Load PHY parameters from a TOML file.
    #[arg(long)]
    params: Option<String>,

    /// CA-SCL list size for Polar decoding (higher = slower, usually better).
    #[arg(long, default_value_t = 16)]
    scl_list_size: usize,

    /// Blind acquisition time window W (seconds). Larger = slower.
    #[arg(long, default_value_t = 0.5)]
    w_sec: f64,

    /// CFAR target false-alarm probability per hypothesis/FFT band.
    #[arg(long, default_value_t = 1e-9)]
    p_fa_total: f64,

    /// Number of RAKE fingers to initialize from acquisition.
    #[arg(long, default_value_t = 3)]
    n_finger: usize,

    /// Ring buffer length in seconds (must be > frame length for continuous decode).
    #[arg(long, default_value_t = 30.0)]
    buffer_sec: f64,

    /// How often to attempt acquisition (milliseconds).
    #[arg(long, default_value_t = 250)]
    acq_interval_ms: u64,

    /// TCP read timeout in milliseconds (0 = no timeout)
    #[arg(long, default_value_t = 0)]
    read_timeout_ms: u64,

    /// Enable verbose debug output (rate-limited).
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn parse_key_hex(s: &str) -> anyhow::Result<[u8; 32]> {
    let b = hex::decode(s).context("invalid hex")?;
    if b.len() != 32 {
        anyhow::bail!("key must be 32 bytes (64 hex chars)");
    }
    let mut k = [0u8; 32];
    k.copy_from_slice(&b);
    Ok(k)
}

fn read_u32_le(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[derive(Clone, Debug)]
struct Pending {
    ti: u64,
    base_abs: u64,
    offsets: Vec<usize>,
    cfo_hz: f64,
    p_max: f32,
    need_abs: u64,
}

struct DebugStats {
    last_dbg: Instant,
    samp: u64,
    pwr_sum: f64,
}

struct AcquisitionState {
    last_acq: Instant,
    last_decoded_ti: Option<u64>,
    search_cursor_ti: Option<u64>,
}

struct SignalBuffer {
    ring: Ring<Complex32>,
    pending: HashMap<u64, Pending>,
    max_pending: usize,
    stash: Vec<u8>,
    io_buf: Vec<u8>,
    raw_buf: Vec<Complex32>,
}

struct ReceiverState {
    t0_s: Option<f64>,
    fs: f64,
    iv_samples: u64,
    acq_guard_samples: u64,
    acq_tail_samples: u64,
    acq: AcquisitionState,
    dbg: DebugStats,
}

struct Receiver {
    args: Args,
    p: Params,
    modem: ScBltcModem,
    signal: SignalBuffer,
    state: ReceiverState,
}

impl Receiver {
    fn new(args: Args) -> anyhow::Result<Self> {
        let p = if let Some(path) = args.params.as_deref() {
            Params::from_file(path)?
        } else {
            Params::default()
        };
        let key = parse_key_hex(&args.key_hex)?;
        let modem = ScBltcModem::new(p.clone(), key)?;

        let fs = p.fs_hz as f64;
        let iv_samples = ((p.fs_hz as f64) * p.iv_res_s).round() as u64;
        let l_sym = p.chip_samples() as u64;
        let ell_last_pilot = 2u64 + 5u64 * (p.n_pilot.saturating_sub(1) as u64);
        let last_pilot_end = (ell_last_pilot + 1) * l_sym;
        let pilot_timing_win: u64 = 32;
        let rake_search_half_samples: u64 =
            ((p.fs_hz as f64) * p.rake_search_half_s).round().max(0.0) as u64;
        let acq_guard_samples =
            last_pilot_end + pilot_timing_win + iv_samples + rake_search_half_samples;
        let acq_tail_samples = acq_guard_samples.saturating_add(iv_samples);

        let cap_samples = (args.buffer_sec.max(1.0) * fs).round() as usize;
        let ring = Ring::<Complex32>::new(cap_samples);

        Ok(Self {
            args,
            p,
            modem,
            signal: SignalBuffer {
                ring,
                pending: HashMap::new(),
                max_pending: 8,
                stash: Vec::new(),
                io_buf: vec![0u8; 32 * 1024],
                raw_buf: Vec::new(),
            },
            state: ReceiverState {
                t0_s: None,
                fs,
                iv_samples,
                acq_guard_samples,
                acq_tail_samples,
                acq: AcquisitionState {
                    last_acq: Instant::now() - Duration::from_secs(3600),
                    last_decoded_ti: None,
                    search_cursor_ti: None,
                },
                dbg: DebugStats {
                    last_dbg: Instant::now() - Duration::from_secs(3600),
                    samp: 0,
                    pwr_sum: 0.0,
                },
            },
        })
    }

    fn accept_connection(&self) -> anyhow::Result<TcpStream> {
        let listener = TcpListener::bind(&self.args.bind)
            .with_context(|| format!("bind {}", self.args.bind))?;
        eprintln!("[rx_tcp] listening on {}", self.args.bind);
        let (stream, peer) = listener.accept().context("accept")?;
        if self.args.read_timeout_ms > 0 {
            stream
                .set_read_timeout(Some(Duration::from_millis(self.args.read_timeout_ms)))
                .context("set_read_timeout")?;
        }
        stream.set_nodelay(true).ok();
        eprintln!("[rx_tcp] connection from {peer}");
        Ok(stream)
    }

    fn handshake(&mut self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let mut magic = [0u8; 8];
        stream.read_exact(&mut magic).context("read magic")?;
        if &magic != MAGIC {
            anyhow::bail!("bad magic: expected {:?}, got {:?}", MAGIC, magic);
        }
        let fs_hz = read_u32_le(stream).context("read fs_hz")?;
        let t0_ns = read_u64_le(stream).context("read t0_ns")?;
        if fs_hz != self.p.fs_hz {
            anyhow::bail!(
                "fs mismatch: sender fs_hz={} but Params.fs_hz={}",
                fs_hz,
                self.p.fs_hz
            );
        }
        eprintln!("[rx_tcp] handshake ok (fs={}Hz, t0_ns={})", fs_hz, t0_ns);
        self.state.t0_s = Some((t0_ns as f64) * 1e-9);
        Ok(())
    }

    fn read_samples(&mut self, stream: &mut TcpStream) -> anyhow::Result<Option<usize>> {
        let n = match stream.read(&mut self.signal.io_buf) {
            Ok(0) => return Ok(None),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(Some(0)),
            Err(e) => return Err(e).context("read tcp stream"),
        };
        self.signal
            .stash
            .extend_from_slice(&self.signal.io_buf[..n]);

        let n_samp = self.signal.stash.len() / 8;
        if n_samp == 0 {
            return Ok(Some(0));
        }
        let used = n_samp * 8;
        let raw = &mut self.signal.raw_buf;
        raw.clear();
        raw.reserve(n_samp);
        for i in 0..n_samp {
            let off = i * 8;
            let re = f32::from_le_bytes(self.signal.stash[off..off + 4].try_into().unwrap());
            let im = f32::from_le_bytes(self.signal.stash[off + 4..off + 8].try_into().unwrap());
            raw.push(Complex32::new(re, im));
        }
        self.signal.stash.drain(..used);
        Ok(Some(n_samp))
    }

    fn push_samples(&mut self, raw: &[Complex32]) {
        self.signal.ring.push_slice(raw);
        if !raw.is_empty() {
            self.state.dbg.samp = self.state.dbg.samp.saturating_add(raw.len() as u64);
            let pwr: f64 = raw
                .iter()
                .map(|v| (v.re as f64) * (v.re as f64) + (v.im as f64) * (v.im as f64))
                .sum();
            self.state.dbg.pwr_sum += pwr;
        }
    }

    fn maybe_log_debug(&mut self) {
        if !self.args.debug {
            return;
        }
        if self.state.dbg.last_dbg.elapsed() < Duration::from_millis(1000) {
            return;
        }
        self.state.dbg.last_dbg = Instant::now();
        let pwr_avg = if self.state.dbg.samp > 0 {
            self.state.dbg.pwr_sum / (self.state.dbg.samp as f64)
        } else {
            0.0
        };
        eprintln!(
            "[rx_tcp][dbg] ring: len={} (base={}, next={}), raw_avg_power={:.3e}, stash_rem_bytes={}, acq_ready={} (acq_tail_samples={})",
            self.signal.ring.len(),
            self.signal.ring.abs_base(),
            self.signal.ring.abs_next(),
            pwr_avg,
            self.signal.stash.len(),
            self.signal.ring.len() >= self.state.acq_tail_samples,
            self.state.acq_tail_samples
        );
        self.state.dbg.samp = 0;
        self.state.dbg.pwr_sum = 0.0;
    }

    fn process_pending(&mut self) -> anyhow::Result<()> {
        if self.signal.pending.is_empty() {
            return Ok(());
        }

        let mut best_ready: Option<Pending> = None;
        for cand in self.signal.pending.values() {
            if self.signal.ring.abs_next() < cand.need_abs {
                continue;
            }
            let better = best_ready
                .as_ref()
                .map(|b| cand.p_max > b.p_max)
                .unwrap_or(true);
            if better {
                best_ready = Some(cand.clone());
            }
        }

        if let Some(cand) = best_ready {
            let hist = self.modem.rrc.taps.len().saturating_sub(1) as u64;
            let start_abs = cand.base_abs.saturating_sub(hist);
            let len = (cand.need_abs - start_abs) as usize;
            if let Some(y) = self.signal.ring.get_vec(start_abs, len) {
                let frame_start_sample = (cand.base_abs - start_abs) as usize;
                let (payload, meta) = self
                    .modem
                    .demod_decode_raw(
                        &y,
                        cand.ti,
                        frame_start_sample,
                        &cand.offsets,
                        cand.cfo_hz,
                        self.args.scl_list_size,
                    )
                    .unwrap_or_else(|_e| (None, sc_bltc::modem::DecodeMeta::error("demod_error")));

                if let Some(pl) = payload {
                    let s = String::from_utf8_lossy(&pl);
                    println!(
                        "[rx_tcp] decoded: {} (crc_ok={}, ver={}, type={}, len={}, ti_tx={}, p_max={:.3e}, cfo={:.2}Hz, offsets={:?})",
                        s,
                        meta.crc_ok,
                        meta.ver,
                        meta.typ,
                        meta.len,
                        cand.ti,
                        cand.p_max,
                        cand.cfo_hz,
                        cand.offsets
                    );
                    self.state.acq.last_decoded_ti = Some(cand.ti);
                } else {
                    eprintln!(
                        "[rx_tcp] decode failed (crc_ok={}, err={:?}, ver={}, type={}, len={}, ti_tx={}, p_max={:.3e}, cfo={:.2}Hz, offsets={:?})",
                        meta.crc_ok,
                        meta.err,
                        meta.ver,
                        meta.typ,
                        meta.len,
                        cand.ti,
                        cand.p_max,
                        cand.cfo_hz,
                        cand.offsets
                    );
                }
            }

            self.signal.pending.remove(&cand.ti);
        }
        Ok(())
    }

    fn run_acquisition(&mut self) -> anyhow::Result<()> {
        if self.state.acq.last_acq.elapsed() < Duration::from_millis(self.args.acq_interval_ms) {
            return Ok(());
        }
        if self.signal.ring.len() <= self.state.acq_tail_samples {
            return Ok(());
        }
        let Some(t0) = self.state.t0_s else {
            return Ok(());
        };

        self.state.acq.last_acq = Instant::now();
        let t_latest = t0
            + ((self
                .signal
                .ring
                .abs_next()
                .saturating_sub(self.state.acq_tail_samples)) as f64)
                / self.state.fs;
        let ti_latest_i = (t_latest / self.p.iv_res_s).floor() as i64;
        let ti_earliest_i = ((t0 + (self.signal.ring.abs_base() as f64) / self.state.fs)
            / self.p.iv_res_s)
            .ceil() as i64;

        let ti_latest = ti_latest_i.max(0) as u64;
        let mut ti_earliest = ti_earliest_i.max(0) as u64;
        if ti_latest < ti_earliest {
            self.state.acq.search_cursor_ti = Some(ti_earliest);
            return Ok(());
        }

        let w_ti = ((self.args.w_sec / self.p.iv_res_s).ceil().max(1.0)) as u64;
        let ti_floor = ti_latest.saturating_sub(w_ti);
        ti_earliest = ti_earliest.max(ti_floor);

        if self.state.acq.search_cursor_ti.is_none() {
            self.state.acq.search_cursor_ti = Some(ti_earliest);
        }
        let mut next_ti = self.state.acq.search_cursor_ti.unwrap();

        if next_ti < ti_earliest {
            if self.args.debug {
                eprintln!(
                    "[rx_tcp][dbg] ring overrun: skipped TI {}..{}",
                    next_ti, ti_earliest
                );
            }
            next_ti = ti_earliest;
        }

        if next_ti > ti_latest {
            self.state.acq.search_cursor_ti = Some(next_ti);
            return Ok(());
        }

        let max_batch = 5000u64;
        let mut ti_end = next_ti + max_batch;
        if ti_end > ti_latest {
            ti_end = ti_latest;
        }

        let n_start_f = ((next_ti as f64) * self.p.iv_res_s - t0) * self.state.fs;
        let n_start = n_start_f.round().max(0.0) as u64;

        if n_start < self.signal.ring.abs_base() {
            let delta_samp = self.signal.ring.abs_base() - n_start;
            let delta_ti = delta_samp.div_ceil(self.state.iv_samples);
            self.state.acq.search_cursor_ti = Some(next_ti.saturating_add(delta_ti));
            return Ok(());
        }

        if n_start.saturating_add(self.state.acq_tail_samples) > self.signal.ring.abs_next() {
            self.state.acq.search_cursor_ti = Some(next_ti);
            return Ok(());
        }

        let max_n_ti_by_end = {
            let slack = self
                .signal
                .ring
                .abs_next()
                .saturating_sub(n_start)
                .saturating_sub(self.state.acq_guard_samples);
            (slack / self.state.iv_samples).max(1)
        };
        let mut n_ti = ti_end - next_ti + 1;
        if n_ti > max_n_ti_by_end {
            n_ti = max_n_ti_by_end;
            ti_end = next_ti + n_ti - 1;
        }
        let n_ti = n_ti as usize;

        let win_len = (n_ti as u64)
            .saturating_mul(self.state.iv_samples)
            .saturating_add(self.state.acq_guard_samples) as usize;

        let Some(y_win) = self.signal.ring.get_vec(n_start, win_len) else {
            if self.args.debug {
                eprintln!(
                    "[rx_tcp][dbg] acq skipped: ring.get_vec failed (n_start={}, win_len={})",
                    n_start, win_len
                );
            }
            self.state.acq.search_cursor_ti = Some(next_ti + 1);
            return Ok(());
        };

        if self.args.debug {
            eprintln!(
                "[rx_tcp][dbg] acq scan: ti={}..{} (n={}), n_start={}, win_len={}",
                next_ti, ti_end, n_ti, n_start, win_len
            );
        }

        let backlog_before_s = (ti_latest.saturating_sub(next_ti) as f64) * self.p.iv_res_s;
        let t_acq0 = Instant::now();
        match self.modem.acquire_fft_raw_window(
            &y_win,
            next_ti,
            n_ti,
            self.args.p_fa_total,
            self.args.n_finger,
        ) {
            Ok(Some(acq)) => {
                if self.state.acq.last_decoded_ti != Some(acq.ti_hat) {
                    let base_abs = n_start + (acq.ti_hat - next_ti) * self.state.iv_samples;
                    let max_off = acq.finger_offsets.iter().copied().max().unwrap_or(acq.n0) as u64;
                    let decode_need_abs = base_abs
                        + max_off
                        + (self.p.frame_samples_with_tail() as u64)
                        + (4 * self.modem.rrc.delay() as u64)
                        + 256;
                    let decode_in_s = (decode_need_abs.saturating_sub(self.signal.ring.abs_next())
                        as f64)
                        / self.state.fs;

                    let cand = Pending {
                        ti: acq.ti_hat,
                        base_abs,
                        offsets: acq.finger_offsets,
                        cfo_hz: acq.cfo_hat_hz,
                        p_max: acq.p_max,
                        need_abs: decode_need_abs,
                    };
                    let update = self
                        .signal
                        .pending
                        .get(&cand.ti)
                        .map(|old| cand.p_max > old.p_max)
                        .unwrap_or(true);
                    if update {
                        if self.args.debug {
                            eprintln!(
                                "[rx_tcp][dbg] acq candidate: ti_hat={}, n0={}, p_max={:.3e}, cfo={:.2}Hz, fingers={:?}, base_abs={} (decode_in~{:.1}s)",
                                acq.ti_hat,
                                acq.n0,
                                acq.p_max,
                                acq.cfo_hat_hz,
                                cand.offsets,
                                base_abs,
                                decode_in_s
                            );
                        } else {
                            eprintln!(
                                "[rx_tcp] acq: ti_hat={}, n0={}, p_max={:.3e}, cfo={:.2}Hz, fingers={:?}, base_abs={} (decode_in~{:.1}s)",
                                acq.ti_hat,
                                acq.n0,
                                acq.p_max,
                                acq.cfo_hat_hz,
                                cand.offsets,
                                base_abs,
                                decode_in_s
                            );
                        }
                        self.signal.pending.insert(cand.ti, cand);
                    }

                    if self.signal.pending.len() > self.signal.max_pending {
                        if let Some((&worst_ti, _)) = self
                            .signal
                            .pending
                            .iter()
                            .min_by(|a, b| a.1.p_max.partial_cmp(&b.1.p_max).unwrap())
                        {
                            self.signal.pending.remove(&worst_ti);
                        }
                    }
                }
            }
            Ok(None) => {
                if self.args.debug {
                    eprintln!("[rx_tcp][dbg] acq none (elapsed={:?})", t_acq0.elapsed());
                }
            }
            Err(e) => {
                eprintln!("[rx_tcp] acq error: {e:#}");
            }
        }
        if self.args.debug {
            let elapsed_s = t_acq0.elapsed().as_secs_f64();
            let span_s = (n_ti as f64) * self.p.iv_res_s;
            let speed_x = if elapsed_s > 0.0 {
                span_s / elapsed_s
            } else {
                f64::INFINITY
            };
            let backlog_after_s =
                (ti_latest.saturating_sub(ti_end.saturating_add(1)) as f64) * self.p.iv_res_s;
            eprintln!(
                "[rx_tcp][dbg] acq realtime: span={:.3}s, elapsed={:.3}s, speed={:.2}x, backlog={:.3}s -> {:.3}s",
                span_s, elapsed_s, speed_x, backlog_before_s, backlog_after_s
            );
        }
        self.state.acq.search_cursor_ti = Some(ti_end + 1);
        Ok(())
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let mut stream = self.accept_connection()?;
        self.handshake(&mut stream)?;

        loop {
            let Some(n_raw) = self.read_samples(&mut stream)? else {
                break;
            };

            if n_raw > 0 {
                let raw_buf = std::mem::take(&mut self.signal.raw_buf);
                self.push_samples(&raw_buf[..n_raw]);
                self.signal.raw_buf = raw_buf;
            }
            self.maybe_log_debug();
            self.process_pending()?;
            self.run_acquisition()?;
        }

        eprintln!("[rx_tcp] connection closed");
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    Receiver::new(Args::parse())?.run()
}
