use anyhow::Context;
use clap::Parser;
use num_complex::Complex32;
use sc_bltc::modem::ScBltcModem;
use sc_bltc::params::Params;
use std::collections::VecDeque;
use std::io;
use std::io::Write;
use std::io::{BufRead, Stdin};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 8] = b"SCBLTC01";

#[derive(Parser, Debug)]
#[command(about = "SC-BLTC sender over TCP (no system audio stack)")]
struct Args {
    /// Receiver address, e.g. 127.0.0.1:5555
    #[arg(long, default_value = "127.0.0.1:5555")]
    addr: String,

    #[arg(
        long,
        default_value = "0000000000000000000000000000000000000000000000000000000000000000"
    )]
    key_hex: String,

    /// Protocol version (Header.Ver)
    #[arg(long, default_value_t = 1)]
    ver: u8,

    /// Message type (Header.Type)
    #[arg(long, default_value_t = 1)]
    typ: u8,

    /// Send N typed frames then exit (0 = infinite).
    #[arg(long, default_value_t = 0)]
    frames: u64,

    /// Initial silence before the first frame (seconds).
    #[arg(long, default_value_t = 0.5)]
    lead_s: f64,

    /// Extra silence between frames (seconds).
    #[arg(long, default_value_t = 0.1)]
    gap_s: f64,

    /// Simulated carrier frequency offset applied to the transmitted baseband (Hz).
    #[arg(long, default_value_t = 0.0)]
    cfo_hz: f64,

    /// Sampling-rate offset between the stream's *actual* sample clock and the declared Fs (ppm).
    /// Positive values send samples slightly faster than Fs (clock runs fast).
    #[arg(long, default_value_t = 0.0)]
    sro_ppm: f64,

    /// Random Doppler stddev (Hz), modeled as a correlated random process added to CFO (0 = off).
    #[arg(long, default_value_t = 0.0)]
    doppler_std_hz: f64,

    /// Doppler correlation time constant (seconds). Smaller = faster variation.
    #[arg(long, default_value_t = 0.5)]
    doppler_tau_s: f64,

    /// Clamp absolute Doppler to this value (Hz). 0 = no clamp.
    #[arg(long, default_value_t = 0.0)]
    doppler_max_hz: f64,

    /// Explicit multipath taps: `delay_samp,gain_db[,phase_deg]`. May be repeated.
    /// If empty, `--mp_paths` can be used to generate random taps.
    #[arg(long)]
    mp_tap: Vec<String>,

    /// Enable random multipath with N paths (0 = off). Includes a direct path at delay 0.
    #[arg(long, default_value_t = 0)]
    mp_paths: usize,

    /// Max extra delay for random multipath taps (samples).
    #[arg(long, default_value_t = 40)]
    mp_max_delay_samp: usize,

    /// Power delay profile slope for random multipath (dB/sample).
    #[arg(long, default_value_t = 0.20)]
    mp_decay_db_per_samp: f32,

    /// AWGN stddev per I/Q component (linear). 0 = no noise.
    #[arg(long, default_value_t = 0.0)]
    noise_std: f32,

    /// Output amplitude scaling.
    #[arg(long, default_value_t = 1.0)]
    amp: f32,

    /// RNG seed (for noise, Doppler/multipath randomness, and random IV offset).
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Stream start time offset from "now" to improve TX/RX time alignment (seconds).
    #[arg(long, default_value_t = 0.2)]
    start_delay_s: f64,

    /// TCP write timeout in milliseconds (0 = no timeout)
    #[arg(long, default_value_t = 0)]
    write_timeout_ms: u64,

    /// Simulated TCP egress delay (milliseconds). Implemented by buffering all outgoing writes
    /// and only releasing them after this delay, to model TX/RX timestamp mismatch.
    #[arg(long, default_value_t = 0)]
    tcp_delay_ms: u64,
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

fn write_u32_le(w: &mut impl Write, x: u32) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn write_u64_le(w: &mut impl Write, x: u64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn unix_now_ns_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn spawn_stdin_lines(stdin: Stdin) -> mpsc::Receiver<Option<String>> {
    let (tx, rx) = mpsc::channel::<Option<String>>();
    std::thread::spawn(move || {
        for line in stdin.lock().lines() {
            match line {
                Ok(s) => {
                    if tx.send(Some(s)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(None); // EOF
    });
    rx
}

struct TcpDelay {
    stream: TcpStream,
    delay: Duration,
    q: VecDeque<(Instant, Vec<u8>)>,
    queued_bytes: usize,
    max_queued_bytes: usize,
}

impl TcpDelay {
    fn new(stream: TcpStream, delay: Duration) -> Self {
        // Safety cap in case the receiver is slow or delay is accidentally huge.
        let max_queued_bytes = 256 * 1024 * 1024;
        Self {
            stream,
            delay,
            q: VecDeque::new(),
            queued_bytes: 0,
            max_queued_bytes,
        }
    }

    fn flush_due(&mut self) -> anyhow::Result<()> {
        if self.delay.is_zero() {
            return Ok(());
        }
        let now = Instant::now();
        while let Some((due, _)) = self.q.front() {
            if *due > now {
                break;
            }
            let (_, bytes) = self.q.pop_front().expect("front exists");
            self.queued_bytes = self.queued_bytes.saturating_sub(bytes.len());
            self.stream
                .write_all(&bytes)
                .context("write tcp (delayed)")?;
        }
        Ok(())
    }

    fn send_bytes(&mut self, bytes: Vec<u8>) -> anyhow::Result<()> {
        if self.delay.is_zero() {
            self.stream.write_all(&bytes).context("write tcp")?;
            return Ok(());
        }

        self.flush_due()?;

        let due = Instant::now() + self.delay;
        self.queued_bytes = self.queued_bytes.saturating_add(bytes.len());
        if self.queued_bytes > self.max_queued_bytes {
            anyhow::bail!(
                "tcp delay queue too large ({} bytes); reduce --tcp_delay_ms or send rate",
                self.queued_bytes
            );
        }
        self.q.push_back((due, bytes));
        Ok(())
    }

    fn drain(&mut self) -> anyhow::Result<()> {
        while let Some((due, _)) = self.q.front() {
            let now = Instant::now();
            if *due > now {
                std::thread::sleep(*due - now);
            }
            self.flush_due()?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Rng64 {
    st: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            st: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.st;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.st = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        ((u as f32) + 1.0) / ((1u32 << 24) as f32 + 2.0)
    }
}

struct Gauss {
    have: bool,
    spare: f32,
}

impl Gauss {
    fn new() -> Self {
        Self {
            have: false,
            spare: 0.0,
        }
    }

    fn next(&mut self, rng: &mut Rng64) -> f32 {
        if self.have {
            self.have = false;
            return self.spare;
        }
        let u1 = rng.next_f32().max(1e-12);
        let u2 = rng.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * th.cos();
        let z1 = r * th.sin();
        self.have = true;
        self.spare = z1;
        z0
    }
}

#[derive(Clone)]
struct MultipathFir {
    taps: Vec<Complex32>,
    dl: Vec<Complex32>,
    pos: usize,
}

impl MultipathFir {
    fn new(taps: Vec<Complex32>) -> anyhow::Result<Self> {
        if taps.is_empty() {
            anyhow::bail!("multipath taps must be non-empty");
        }
        Ok(Self {
            dl: vec![Complex32::new(0.0, 0.0); taps.len()],
            taps,
            pos: 0,
        })
    }

    fn filter(&mut self, x: Complex32) -> Complex32 {
        let l = self.taps.len();
        self.dl[self.pos] = x;
        let mut y = Complex32::new(0.0, 0.0);
        for k in 0..l {
            let idx = (self.pos + l - k) % l;
            y += self.taps[k] * self.dl[idx];
        }
        self.pos += 1;
        if self.pos >= l {
            self.pos = 0;
        }
        y
    }
}

struct DopplerOu {
    f_hz: f64,
    a: f64,
    b: f64,
    max_abs_hz: f64,
}

impl DopplerOu {
    fn new(std_hz: f64, tau_s: f64, fs_hz: f64, max_abs_hz: f64) -> Self {
        let a = if tau_s > 0.0 {
            (-1.0 / (fs_hz * tau_s)).exp()
        } else {
            0.0
        };
        let b = std_hz * (1.0 - a * a).max(0.0).sqrt();
        Self {
            f_hz: 0.0,
            a,
            b,
            max_abs_hz: max_abs_hz.max(0.0),
        }
    }

    fn step_hz(&mut self, rng: &mut Rng64, gauss: &mut Gauss) -> f64 {
        let w = gauss.next(rng) as f64;
        self.f_hz = self.a * self.f_hz + self.b * w;
        if self.max_abs_hz > 0.0 {
            self.f_hz = self.f_hz.clamp(-self.max_abs_hz, self.max_abs_hz);
        }
        self.f_hz
    }
}

struct ChannelState {
    phi: f32,
    mp: Option<MultipathFir>,
    doppler: Option<DopplerOu>,
}

fn normalize_taps(mut taps: Vec<Complex32>) -> Vec<Complex32> {
    let e: f32 = taps.iter().map(|c| c.norm_sqr()).sum();
    if e > 0.0 {
        let s = 1.0 / e.sqrt();
        for t in &mut taps {
            *t *= s;
        }
    }
    taps
}

fn parse_mp_tap(s: &str, rng: &mut Rng64) -> anyhow::Result<(usize, Complex32)> {
    let ss = s.trim().replace(':', ",");
    let parts: Vec<&str> = ss
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .collect();
    if parts.len() < 2 || parts.len() > 3 {
        anyhow::bail!("mp_tap must be `delay_samp,gain_db[,phase_deg]`, got `{s}`");
    }
    let delay: usize = parts[0].parse().context("parse delay_samp")?;
    let gain_db: f32 = parts[1].parse().context("parse gain_db")?;
    let phase_rad: f32 = if parts.len() == 3 {
        let deg: f32 = parts[2].parse().context("parse phase_deg")?;
        deg * (std::f32::consts::PI / 180.0)
    } else {
        2.0 * std::f32::consts::PI * rng.next_f32()
    };
    let gain_lin = 10.0f32.powf(gain_db / 20.0);
    Ok((delay, Complex32::from_polar(gain_lin, phase_rad)))
}

fn build_random_multipath(
    paths: usize,
    max_delay: usize,
    decay_db_per_samp: f32,
    rng: &mut Rng64,
    gauss: &mut Gauss,
) -> Vec<Complex32> {
    if paths <= 1 || max_delay == 0 {
        return vec![Complex32::new(1.0, 0.0)];
    }
    let mut entries: Vec<(usize, Complex32)> = vec![(0, Complex32::new(1.0, 0.0))];
    for _ in 1..paths {
        let d = 1 + (rng.next_u64() as usize) % max_delay;
        let pwr = 10.0f32.powf(-decay_db_per_samp * (d as f32) / 10.0);
        let sigma = (pwr / 2.0).sqrt();
        let gr = sigma * gauss.next(rng);
        let gi = sigma * gauss.next(rng);
        entries.push((d, Complex32::new(gr, gi)));
    }
    let max_d = entries.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let mut taps = vec![Complex32::new(0.0, 0.0); max_d + 1];
    for (d, g) in entries {
        taps[d] += g;
    }
    normalize_taps(taps)
}

#[allow(clippy::too_many_arguments)]
fn apply_channel_sample(
    x: Complex32,
    amp: f32,
    cfo_hz: f64,
    fs_actual_hz: f64,
    noise_std: f32,
    st: &mut ChannelState,
    rng: &mut Rng64,
    gauss: &mut Gauss,
) -> Complex32 {
    let mut y = if let Some(mp) = &mut st.mp {
        mp.filter(x)
    } else {
        x
    };

    let mut f_hz = cfo_hz;
    if let Some(d) = &mut st.doppler {
        f_hz += d.step_hz(rng, gauss);
    }
    if f_hz != 0.0 {
        y *= Complex32::from_polar(1.0, st.phi);
        let dphi = (2.0 * std::f64::consts::PI * f_hz / fs_actual_hz) as f32;
        st.phi += dphi;
        if st.phi.abs() > 1000.0 {
            st.phi = st.phi.rem_euclid(2.0 * std::f32::consts::PI);
        }
    }

    y *= amp;

    if noise_std > 0.0 {
        let nr = noise_std * gauss.next(rng);
        let ni = noise_std * gauss.next(rng);
        y += Complex32::new(nr, ni);
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn send_silence_samples(
    tx: &mut TcpDelay,
    buf: &mut [Complex32],
    args: &Args,
    ch: &mut ChannelState,
    rng: &mut Rng64,
    gauss: &mut Gauss,
    fs_actual_hz: f64,
    stream_t0_inst: Instant,
    n_sent: &mut u64,
    n_total: usize,
) -> anyhow::Result<()> {
    let mut left = n_total;
    while left > 0 {
        let n = left.min(buf.len());
        for v in &mut buf[..n] {
            *v = apply_channel_sample(
                Complex32::new(0.0, 0.0),
                args.amp,
                args.cfo_hz,
                fs_actual_hz,
                args.noise_std,
                ch,
                rng,
                gauss,
            );
        }
        send_block_paced(tx, &buf[..n], fs_actual_hz, stream_t0_inst, n_sent)?;
        left -= n;
    }
    Ok(())
}

fn send_block_paced(
    tx: &mut TcpDelay,
    xs: &[Complex32],
    fs_actual_hz: f64,
    t0_inst: Instant,
    n_sent: &mut u64,
) -> anyhow::Result<()> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for &s in xs {
        out.extend_from_slice(&s.re.to_le_bytes());
        out.extend_from_slice(&s.im.to_le_bytes());
    }
    tx.send_bytes(out).context("queue samples")?;
    *n_sent += xs.len() as u64;

    let target = Duration::from_secs_f64((*n_sent as f64) / fs_actual_hz);
    if let Some(remaining) = target.checked_sub(t0_inst.elapsed()) {
        if remaining > Duration::from_millis(0) {
            std::thread::sleep(remaining);
        }
    }

    // Opportunistically flush any packets whose delay has elapsed.
    tx.flush_due().ok();
    Ok(())
}

struct Transmitter {
    args: Args,
    p: Params,
    modem: ScBltcModem,
}

impl Transmitter {
    fn new(args: Args) -> anyhow::Result<Self> {
        let p = Params::default();
        let key = parse_key_hex(&args.key_hex)?;
        let modem = ScBltcModem::new(p.clone(), key)?;
        Ok(Self { args, p, modem })
    }

    fn connect_stream(&self) -> anyhow::Result<TcpStream> {
        let stream = TcpStream::connect(&self.args.addr)
            .with_context(|| format!("connect {}", self.args.addr))?;
        if self.args.write_timeout_ms > 0 {
            stream
                .set_write_timeout(Some(Duration::from_millis(self.args.write_timeout_ms)))
                .context("set_write_timeout")?;
        }
        stream.set_nodelay(true).ok();
        Ok(stream)
    }

    fn write_handshake(&self, stream: &mut TcpStream) -> anyhow::Result<u64> {
        stream.write_all(MAGIC).context("write magic")?;
        write_u32_le(stream, self.p.fs_hz).context("write fs_hz")?;
        stream.flush().ok();

        eprintln!(
            "[tx_tcp] connected to {} (fs={}Hz), streaming I/Q f32 LE continuously (tcp_delay_ms={})",
            self.args.addr, self.p.fs_hz, self.args.tcp_delay_ms
        );

        let fs_u64 = self.p.fs_hz as u64;
        let ts_ns = 1_000_000_000u64 / fs_u64;
        let now_ns = unix_now_ns_u64();
        let min_future_ns = now_ns.saturating_add(5_000_000);
        let stream_t0_wall_ns = min_future_ns.div_ceil(ts_ns) * ts_ns;
        write_u64_le(stream, stream_t0_wall_ns).context("write t0_ns")?;
        stream.flush().ok();
        Ok(stream_t0_wall_ns)
    }

    fn build_channel(
        &self,
        rng: &mut Rng64,
        gauss: &mut Gauss,
        fs_actual_hz: f64,
        iv_samples: u64,
    ) -> anyhow::Result<ChannelState> {
        let mp = if !self.args.mp_tap.is_empty() {
            let mut max_d = 0usize;
            let mut min_d = usize::MAX;
            let mut has_d0 = false;
            let mut entries: Vec<(usize, Complex32)> = Vec::with_capacity(self.args.mp_tap.len());
            for s in &self.args.mp_tap {
                let (d, g) = parse_mp_tap(s, rng)?;
                max_d = max_d.max(d);
                min_d = min_d.min(d);
                has_d0 |= d == 0;
                entries.push((d, g));
            }
            if !has_d0 {
                eprintln!(
                    "[tx_tcp][warn] mp_tap has no delay-0 direct path; earliest tap at {} samples (IV={} samples). \
                     If you intended multipath, repeat --mp-tap and include e.g. --mp-tap 0,0.",
                    min_d,
                    iv_samples
                );
            }
            let mut taps = vec![Complex32::new(0.0, 0.0); max_d + 1];
            for (d, g) in entries {
                taps[d] += g;
            }
            Some(MultipathFir::new(normalize_taps(taps))?)
        } else if self.args.mp_paths > 0 {
            Some(MultipathFir::new(build_random_multipath(
                self.args.mp_paths,
                self.args.mp_max_delay_samp,
                self.args.mp_decay_db_per_samp,
                rng,
                gauss,
            ))?)
        } else {
            None
        };

        let doppler = if self.args.doppler_std_hz > 0.0 {
            let mut d = DopplerOu::new(
                self.args.doppler_std_hz,
                self.args.doppler_tau_s,
                fs_actual_hz,
                self.args.doppler_max_hz,
            );
            d.f_hz = self.args.doppler_std_hz * (gauss.next(rng) as f64);
            if d.max_abs_hz > 0.0 {
                d.f_hz = d.f_hz.clamp(-d.max_abs_hz, d.max_abs_hz);
            }
            Some(d)
        } else {
            None
        };

        Ok(ChannelState {
            phi: 0.0,
            mp,
            doppler,
        })
    }

    fn wait_for_start(&self, stream_t0_wall_ns: u64) {
        loop {
            let now = unix_now_ns_u64();
            if now >= stream_t0_wall_ns {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn run(&self) -> anyhow::Result<()> {
        if self.args.start_delay_s > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(self.args.start_delay_s.max(0.0)));
        }

        let mut stream = self.connect_stream()?;
        let stream_t0_wall_ns = self.write_handshake(&mut stream)?;

        let fs_nom_hz = self.p.fs_hz as f64;
        let fs_actual_hz = fs_nom_hz * (1.0 + self.args.sro_ppm * 1e-6);
        if fs_actual_hz <= 0.0 {
            anyhow::bail!("fs_actual_hz must be positive (check --sro_ppm)");
        }
        let iv_samples = (fs_nom_hz * self.p.iv_res_s).round() as u64;

        let mut tx = TcpDelay::new(stream, Duration::from_millis(self.args.tcp_delay_ms));

        let mut rng = Rng64::new(self.args.seed);
        let mut gauss = Gauss::new();

        let stream_t0_wall = (stream_t0_wall_ns as f64) * 1e-9;
        self.wait_for_start(stream_t0_wall_ns);
        let stream_t0_inst = Instant::now();
        let mut n_sent: u64 = 0;

        let mut ch = self.build_channel(&mut rng, &mut gauss, fs_actual_hz, iv_samples)?;

        eprintln!(
            "[tx_tcp] channel: fs_actual={:.3}Hz (sro_ppm={:+.3}), mp={}, doppler_std_hz={:.3}",
            fs_actual_hz,
            self.args.sro_ppm,
            if ch.mp.is_some() { "on" } else { "off" },
            self.args.doppler_std_hz
        );

        let mut silence_buf = vec![Complex32::new(0.0, 0.0); 1024];
        if self.args.lead_s > 0.0 {
            let n0 = (self.args.lead_s * fs_actual_hz).round() as usize;
            send_silence_samples(
                &mut tx,
                &mut silence_buf,
                &self.args,
                &mut ch,
                &mut rng,
                &mut gauss,
                fs_actual_hz,
                stream_t0_inst,
                &mut n_sent,
                n0,
            )?;
        }

        eprintln!("[tx_tcp] type a line and press Enter to send (Ctrl-D to quit)");
        let stdin_rx = spawn_stdin_lines(std::io::stdin());
        let mut stdin_eof = false;
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();

        let mut frame_idx: u64 = 0;
        loop {
            if self.args.frames != 0 && frame_idx >= self.args.frames {
                break;
            }

            while let Ok(m) = stdin_rx.try_recv() {
                match m {
                    Some(s) => pending.push_back(s.into_bytes()),
                    None => stdin_eof = true,
                }
            }

            let payload = if let Some(mut b) = pending.pop_front() {
                b.truncate(26); // Spec §3.A0.
                b
            } else {
                if stdin_eof {
                    break;
                }
                // Keep streaming silence while waiting for user input.
                let n = silence_buf.len();
                send_silence_samples(
                    &mut tx,
                    &mut silence_buf,
                    &self.args,
                    &mut ch,
                    &mut rng,
                    &mut gauss,
                    fs_actual_hz,
                    stream_t0_inst,
                    &mut n_sent,
                    n,
                )?;
                continue;
            };

            // Spec §4.B.1.
            let t_next_sample = stream_t0_wall + (n_sent as f64) / fs_actual_hz;
            let t_base = t_next_sample + self.args.gap_s.max(0.0);
            let mut ti = (t_base / self.p.iv_res_s).floor() as u64 + 1;
            let n_off = rng.next_u64() % iv_samples;
            let mut t_frame = (ti as f64) * self.p.iv_res_s + (n_off as f64) / fs_nom_hz;
            if t_frame < t_base {
                ti += 1;
                t_frame = (ti as f64) * self.p.iv_res_s + (n_off as f64) / fs_nom_hz;
            }

            let n_target = ((t_frame - stream_t0_wall) * fs_actual_hz).round() as i64;
            let n_sil = (n_target - (n_sent as i64)).max(0) as usize;
            if n_sil > 0 {
                send_silence_samples(
                    &mut tx,
                    &mut silence_buf,
                    &self.args,
                    &mut ch,
                    &mut rng,
                    &mut gauss,
                    fs_actual_hz,
                    stream_t0_inst,
                    &mut n_sent,
                    n_sil,
                )?;
            }

            let frame = self
                .modem
                .build_frame_samples(&payload, self.args.ver, self.args.typ, Some(t_frame))
                .context("build_frame_samples")?;

            eprintln!(
                "[tx_tcp] frame={} ti_tx={} iv_off_samp={} t_frame={:.6}",
                frame_idx, frame.ti_tx, n_off, t_frame
            );

            let mut tmp = Vec::with_capacity(2048);
            for &x in &frame.samples {
                tmp.push(apply_channel_sample(
                    x,
                    self.args.amp,
                    self.args.cfo_hz,
                    fs_actual_hz,
                    self.args.noise_std,
                    &mut ch,
                    &mut rng,
                    &mut gauss,
                ));
                if tmp.len() >= 2048 {
                    send_block_paced(&mut tx, &tmp, fs_actual_hz, stream_t0_inst, &mut n_sent)?;
                    tmp.clear();
                }
            }
            if !tmp.is_empty() {
                send_block_paced(&mut tx, &tmp, fs_actual_hz, stream_t0_inst, &mut n_sent)?;
            }

            frame_idx += 1;
        }

        tx.drain().ok();
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    Transmitter::new(Args::parse())?.run()
}
