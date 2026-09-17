# Routing-aware wavelength reassignment — hardware feasibility and mechanism

Scope: is reassignment possible in the Section 3 design (wafer-scale SiN interposer, X–Y
waveguide grid, O-E-O relay at row/column intersections, per die 12 waveguides x 32 lambda,
375–384 Tx + 375–384 Rx rings), and if so how / when / what it looks like.

## 1. What a "link" is physically
- Each die owns its egress waveguides: 12 x 32 lambda = 384 lambda, taken as 6 waveguides along
  the die's row and 6 along its column (192 lambda per direction). This is the only reading of
  Section 3 consistent with the htsim row/column mesh (every GPU has a direct link to each of
  the 3 other GPUs in its row and the 3 in its column).
- A source's row waveguide passes under the 3 other reticles of the row. A wavelength lambda_k
  on that waveguide reaches whichever reticle has a drop ring ON RESONANCE at lambda_k; all
  other reticles' rings for lambda_k are parked off-resonance. So "the link G0->G3 has 64
  lambda" means: G3 has 64 drop rings on resonance on G0's row waveguides, G1 and G2 have
  their rings for those 64 wavelengths parked.
- Two-hop paths (row then column) go through an O-E-O relay at the intersection reticle: the
  relay drops the data, and re-modulates it on ITS OWN column waveguides. Relay traffic
  therefore consumes the intermediate reticle's column budget, which the allocation's load
  accounting includes (hops()).
- Uniform allocation: 192 / 3 = 64 lambda per link = 64 x 32 Gb/s = 2.05 Tb/s = 256 GB/s per
  link. NOTE: the htsim wafer runs use 1800 GB/s per intra link (a Glass-FB RDL number), i.e.
  7x the Section 3 optical budget in aggregate (6 x 1800 = 10.8 TB/s per die vs 1.5 TB/s).
  Reassignment must be evaluated at the architecture-consistent 256 GB/s per link as well.

## 2. Reassignment = receiver-side ring park/un-park + transmitter-side lane re-map
- Tx side: nothing optical changes. Each Tx ring modulates a fixed lambda_k on a fixed
  waveguide; the EIC decides which destination's data lanes drive which modulators (an
  electrical crossbar in the SerDes/link layer that already exists for lane mapping).
- Rx side: to move m wavelengths from destination k to destination j on the same source
  bus, park k's m drop rings for those wavelengths and un-park m spare rings at j.
  "Park" = shift the resonance ~1 nm off the channel: with FWHM 0.19 nm that is ~5 linewidths,
  a Lorentzian drop of about -20 dB, i.e. the parked ring is transparent to lambda_k.
  The ferroelectric device's ~1 nm non-volatile range is exactly this park/un-park stroke;
  no ring ever has to hop across the 32-channel comb (which would need FSR-scale tuning).
- Ring inventory: flexibility requires spare (parked) Rx rings at every destination for the
  wavelengths it might be assigned. With a cap of 2x the uniform share per source link, a
  destination needs at most 2 x 64 = 128 rings per source bus; in the LLaMA example the
  demand-aware allocation activates at most 543 of 768 provisioned rings (1.41x the uniform
  384). Parked rings are free only because the tuner is non-volatile: with thermo-optic
  tuning every parked ring would need a heater bias (1–3 mW) to stay off its channel, i.e.
  0.4–1.2 W per die for 384 spare rings; with the ferroelectric setpoint the parked state
  costs 0 W and one write.
- Optical budget check per waveguide: 32 lambda x 32 Gb/s = 1.024 Tb/s, unchanged; reassignment
  never over-subscribes a waveguide, it only changes who drops each wavelength. Drop-ring
  through-loss and crosstalk are the same as in the uniform design because the number of ON
  rings per wavelength on a bus is always exactly one.

## 3. When to reassign (routing epoch)
- Input: per-expert token counts per layer, which the gate already computes every step for the
  load-balancing loss. Expert specialization is persistent (Fig. 1), so the demand matrix
  drifts slowly.
- Trigger: at an iteration boundary (after the optimizer step, before the next forward), if
  the allocation recomputed from the last E iterations' token counts differs from the current
  one by >= 8 lambda (one waveguide-quarter) on any link. E ~ 100–1000 iterations.
- Cost of a reassignment: only rings whose state changes are written (tens per die); writes
  are ~10 us each and independent per ring, so a reassignment fits inside one iteration
  boundary and adds no per-round stall. Write budget: with epochs of >= 100 iterations, a
  10^6-iteration run performs <= 10^4 reassignments per ring, well below 10^7 endurance.
- Cold start: allocate uniformly (all 384 rings/die on their default wavelengths), measure for
  E iterations, then reassign.

## 4. What the network model evaluates (htsim_tcp_wafer -lambda-alloc demand)
- Per source, the 6 outgoing intra-wafer links share a fixed budget (6 x link rate); each link
  gets a share proportional to the demand routed over it (hops-aware), floor 0.1x, cap 2x.
- -alloc-matrix <stale> = allocation computed from an outdated demand (robustness to drift);
  -alloc-cap = ring over-provisioning factor; -alloc-inter = also split gateway budget.
- Runs needed: uniform vs demand vs stale, at intra 1800 GB/s (current) AND 256 GB/s
  (architecture-consistent), with and without the thermal stall.

## 5. Caveats to state in the paper
- EP-group placement: an EP group spans tp x ep GPUs = 32 (Mixtral: 2 wafers), 16 (LLaMA: 1
  wafer), 64 (Qwen: 4 wafers). Section 4.4's claim that EP <= 16 keeps all-to-all inside one
  wafer holds only for LLaMA; for Mixtral half of every all-to-all crosses a gateway. Intra-wafer
  reassignment cannot fix gateway-limited traffic; -alloc-inter addresses the gateway share.
- Reassignment granularity is one wavelength (32 Gb/s); the model uses continuous rates.
- The model keeps the per-source budget fixed but lets a destination's ON-ring count vary; the
  cap bounds it. A destination-side ring budget (e.g. exactly 768 rings) is a further
  constraint not enforced in the current model (reported as max rings needed instead).
