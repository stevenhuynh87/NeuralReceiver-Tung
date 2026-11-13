"""Link-level simulation and data batch generation module.

Provides physical layer configuration (LinkConfig) and batch data generation
(batch_generator) producing transmit bits, received OFDM grids, pilot symbols,
noise PSD, and channel frequency responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from src.config import load_config
from src.sim.channels import SyntheticCDL, SyntheticCDLChannel, make_cdl

try:
    import sionna
    import sionna.phy
    from sionna.phy.channel import OFDMChannel
    from sionna.phy.mapping import BinarySource, Mapper
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
    from sionna.phy.utils import ebnodb2no, expand_to_rank

    _SIONNA_AVAILABLE = True
except ImportError:
    try:
        import sionna
        from sionna.channel import OFDMChannel  # type: ignore
        from sionna.mapping import BinarySource, Mapper  # type: ignore
        from sionna.ofdm import ResourceGrid, ResourceGridMapper  # type: ignore
        from sionna.utils import ebnodb2no, expand_to_rank  # type: ignore

        _SIONNA_AVAILABLE = True
    except ImportError:
        _SIONNA_AVAILABLE = False


class BatchData(NamedTuple):
    """5-element tuple container for batch generator outputs."""

    b: torch.Tensor
    Y: torch.Tensor
    X_pilot: torch.Tensor
    N0: Union[float, torch.Tensor]
    H: torch.Tensor


class SyntheticPilotPattern:
    """Fallback pilot pattern when Sionna is unavailable."""

    def __init__(self, num_pilots: int = 0, device: Optional[Union[str, torch.device]] = None):
        dev = str(device) if device is not None else "cpu"
        self.pilots = torch.ones((1, 1, num_pilots), dtype=torch.complex64, device=dev)


class SyntheticResourceGrid:
    """Fallback resource grid when Sionna is unavailable."""

    def __init__(self, config: "LinkConfig", device: Optional[Union[str, torch.device]] = None):
        self.num_ofdm_symbols = config.num_ofdm_symbols
        self.fft_size = config.fft_size
        self.subcarrier_spacing = config.subcarrier_spacing
        self.num_tx = config.num_tx_ant
        self.num_rx = config.num_rx_ant
        self.dc_null = config.dc_null
        self.num_guard_carriers = config.num_guard_carriers
        self.pilot_pattern = SyntheticPilotPattern(num_pilots=config.num_pilot_symbols, device=device)

        guards = sum(config.num_guard_carriers) if config.num_guard_carriers else 0
        dc = 1 if config.dc_null else 0
        self.num_effective_subcarriers = config.fft_size - guards - dc

        num_pilot_syms = len(config.pilot_ofdm_symbol_indices)
        num_data_ofdm_syms = config.num_ofdm_symbols - num_pilot_syms
        self.num_data_symbols = self.num_effective_subcarriers * num_data_ofdm_syms


@dataclass
class LinkConfig:
    """Physical layer and OFDM link configuration.

    Parameters
    ----------
    fft_size : int
        Number of subcarriers forming the resource grid (default: 128).
    num_ofdm_symbols : int
        Number of OFDM symbols per slot (default: 14).
    subcarrier_spacing : float
        Subcarrier spacing in Hz (default: 30000.0).
    carrier_frequency : float
        Carrier frequency in Hz (default: 3.5e9).
    num_tx_ant : int
        Number of transmit antenna elements (default: 1).
    num_rx_ant : int
        Number of receive antenna elements (default: 2).
    pilot_ofdm_symbol_indices : List[int]
        Indices of OFDM symbols carrying pilots (default: [2, 11]).
    num_guard_carriers : List[int]
        Guard subcarriers [left, right] (default: [5, 6]).
    dc_null : bool
        Whether DC subcarrier is nulled (default: True).
    num_bits_per_symbol : int
        Number of bits per constellation symbol (default: 6 for 64-QAM).
    coderate : float
        Channel code rate (default: 0.9).
    pilot_pattern : str
        Pilot pattern scheme ('kronecker') (default: 'kronecker').
    cyclic_prefix_length : int
        Cyclic prefix length (default: 0).
    num_streams_per_tx : int
        Spatial streams per transmitter (default: 1).
    modulation_type : str
        Modulation scheme name ('qam', '64QAM', etc.) (default: 'qam').
    cdl_model : str
        Channel CDL profile model ('A', 'B', 'C', 'D') (default: 'C').
    delay_spread : float
        RMS delay spread in seconds (default: 1.0e-7).
    speed : float
        User mobility speed in m/s (default: 0.0).
    seed : int
        Default random seed (default: 42).
    """

    fft_size: int = 128
    num_ofdm_symbols: int = 14
    subcarrier_spacing: float = 30000.0
    carrier_frequency: float = 3.5e9
    num_tx_ant: int = 1
    num_rx_ant: int = 2
    pilot_ofdm_symbol_indices: List[int] = field(default_factory=lambda: [2, 11])
    num_guard_carriers: List[int] = field(default_factory=lambda: [5, 6])
    dc_null: bool = True
    num_bits_per_symbol: int = 6
    coderate: float = 0.9
    pilot_pattern: str = "kronecker"
    cyclic_prefix_length: int = 0
    num_streams_per_tx: int = 1
    modulation_type: str = "qam"
    cdl_model: str = "C"
    delay_spread: float = 1.0e-7
    speed: float = 0.0
    seed: int = 42

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> LinkConfig:
        """Create LinkConfig instance from single-source YAML configuration dictionary.

        If cfg is None, loads configuration automatically via load_config().
        """
        if cfg is None:
            cfg = load_config()

        phy = cfg.get("phy", {})
        chan = cfg.get("channel", {})
        mod = phy.get("modulation", {})
        if isinstance(mod, dict):
            num_bits = int(mod.get("num_bits_per_symbol", 6))
            mod_type = str(mod.get("type", "qam"))
        else:
            num_bits = 6
            mod_type = "qam"

        speed = chan.get("speed", chan.get("speed_min", 0.0))

        return cls(
            fft_size=int(phy.get("fft_size", 128)),
            num_ofdm_symbols=int(phy.get("num_ofdm_symbols", 14)),
            subcarrier_spacing=float(phy.get("subcarrier_spacing", 30000.0)),
            carrier_frequency=float(phy.get("carrier_frequency", 3.5e9)),
            num_tx_ant=int(phy.get("num_tx_ant", phy.get("num_tx_antennas", 1))),
            num_rx_ant=int(phy.get("num_rx_ant", phy.get("num_rx_antennas", 2))),
            pilot_ofdm_symbol_indices=list(phy.get("pilot_ofdm_symbol_indices", [2, 11])),
            num_guard_carriers=list(phy.get("num_guard_carriers", phy.get("guard_subcarriers", [5, 6]))),
            dc_null=bool(phy.get("dc_null", True)),
            num_bits_per_symbol=num_bits,
            coderate=float(phy.get("coderate", 0.9)),
            pilot_pattern=str(phy.get("pilot_pattern", "kronecker")),
            cyclic_prefix_length=int(phy.get("cyclic_prefix_length", 0)),
            num_streams_per_tx=1,
            modulation_type=mod_type,
            cdl_model=str(chan.get("cdl_model", "C")),
            delay_spread=float(chan.get("delay_spread", 1.0e-7)),
            speed=float(speed),
            seed=int(cfg.get("seed", 42)),
        )

    def create_resource_grid(self, device: Optional[Union[str, torch.device]] = None) -> Any:
        """Create and return a Sionna ResourceGrid or synthetic equivalent."""
        if _SIONNA_AVAILABLE:
            try:
                import sionna
                if hasattr(sionna, "phy") and hasattr(sionna.phy, "config"):
                    sionna.phy.config.seed = int(self.seed)
                elif hasattr(sionna, "config"):
                    sionna.config.seed = int(self.seed)
            except Exception:
                pass
            return ResourceGrid(
                num_ofdm_symbols=self.num_ofdm_symbols,
                fft_size=self.fft_size,
                subcarrier_spacing=self.subcarrier_spacing,
                num_tx=self.num_tx_ant,
                num_streams_per_tx=self.num_streams_per_tx,
                cyclic_prefix_length=self.cyclic_prefix_length,
                dc_null=self.dc_null,
                pilot_pattern=self.pilot_pattern,
                pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices,
                num_guard_carriers=self.num_guard_carriers,
            )
        return SyntheticResourceGrid(self, device=device)

    @property
    def num_tx_antennas(self) -> int:
        """Alias for num_tx_ant."""
        return self.num_tx_ant

    @property
    def num_rx_antennas(self) -> int:
        """Alias for num_rx_ant."""
        return self.num_rx_ant

    @property
    def guard_subcarriers(self) -> List[int]:
        """Alias for num_guard_carriers."""
        return self.num_guard_carriers

    @property
    def num_effective_subcarriers(self) -> int:
        """Number of active subcarriers excluding guard bands and DC null."""
        guards = sum(self.num_guard_carriers) if self.num_guard_carriers else 0
        dc = 1 if self.dc_null else 0
        return self.fft_size - guards - dc

    @property
    def num_pilot_symbols(self) -> int:
        """Total number of pilot symbols allocated per slot."""
        return self.num_effective_subcarriers * len(self.pilot_ofdm_symbol_indices)

    @property
    def num_data_symbols(self) -> int:
        """Total number of data symbols allocated per slot."""
        if _SIONNA_AVAILABLE:
            rg = self.create_resource_grid()
            return int(rg.num_data_symbols)
        num_pilot_syms = len(self.pilot_ofdm_symbol_indices)
        num_data_ofdm_syms = self.num_ofdm_symbols - num_pilot_syms
        return self.num_effective_subcarriers * num_data_ofdm_syms

    @property
    def num_data_bits(self) -> int:
        """Codeword length n = num_data_symbols * num_bits_per_symbol."""
        return int(self.num_data_symbols * self.num_bits_per_symbol)

    @property
    def num_info_bits(self) -> int:
        """Number of information bits k = int(n * coderate)."""
        return int(self.num_data_bits * self.coderate)

    @property
    def n(self) -> int:
        """Codeword length alias n."""
        return self.num_data_bits

    @property
    def k(self) -> int:
        """Information bit length alias k."""
        return self.num_info_bits


def batch_generator(
    channel: Optional[Any] = None,
    link_config: Optional[LinkConfig] = None,
    batch_size: int = 16,
    ebno_db: Union[float, torch.Tensor] = 10.0,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    as_generator: bool = False,
) -> Union[BatchData, Generator[BatchData, None, None]]:
    """Generate transmit-channel-receive batches for OFDM transmission.

    Parameters
    ----------
    channel : CDL or SyntheticCDLChannel, optional
        Sionna CDL or SyntheticCDL channel model. Created via make_cdl() if None.
    link_config : LinkConfig, optional
        Physical layer link configuration. Created via LinkConfig.from_config() if None.
    batch_size : int, optional
        Number of OFDM slots per batch (default: 16).
    ebno_db : float or torch.Tensor, optional
        Signal-to-noise ratio Eb/N0 in dB (default: 10.0).
    seed : int, optional
        Deterministic random seed.
    device : str or torch.device, optional
        Target computation device ('cpu', 'cuda').
    as_generator : bool, optional
        If True, returns an infinite Python generator yielding BatchData tuples.
        If False (default), generates and returns a single BatchData tuple immediately.

    Returns
    -------
    BatchData or Generator[BatchData, None, None]
        NamedTuple (b, Y, X_pilot, N0, H) containing:
        - b: Transmit bits tensor (batch_size, num_tx, 1, num_data_bits)
        - Y: Received post-FFT frequency grid (batch_size, num_rx_ant, num_ofdm_symbols, fft_size)
        - X_pilot: Known pilot symbols tensor
        - N0: Noise power spectral density scalar float > 0
        - H: Perfect CSI channel frequency response (batch_size, num_rx_ant, num_tx, num_ofdm_symbols, fft_size)
    """
    # Polymorphic argument support: if link_config was passed as first arg
    if isinstance(channel, LinkConfig):
        link_config = channel
        channel = None

    if link_config is None:
        link_config = LinkConfig.from_config()

    if device is None:
        if channel is not None and hasattr(channel, "device"):
            device = getattr(channel, "device")
        else:
            cfg = load_config()
            preferred_device = cfg.get("training", {}).get("device", "cuda")
            device = preferred_device if torch.cuda.is_available() and "cuda" in str(preferred_device).lower() else "cpu"

    if channel is None:
        channel = make_cdl(
            model=link_config.cdl_model,
            v=link_config.speed,
            delay_spread=link_config.delay_spread,
            seed=seed if seed is not None else link_config.seed,
            carrier_frequency=link_config.carrier_frequency,
            rx_ant=link_config.num_rx_ant,
            tx_ant=link_config.num_tx_ant,
            device=device,
        )

    def _generate_single_batch() -> BatchData:
        if seed is not None:
            if hasattr(channel, "reset_seed"):
                channel.reset_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        if _SIONNA_AVAILABLE:
            rg = link_config.create_resource_grid(device=device)
            if seed is not None:
                try:
                    import sionna
                    if hasattr(sionna, "phy") and hasattr(sionna.phy, "config"):
                        sionna.phy.config.seed = seed
                    elif hasattr(sionna, "config"):
                        sionna.config.seed = seed
                except Exception:
                    pass
            bs = BinarySource(seed=seed)
            mapper = Mapper("qam", link_config.num_bits_per_symbol)
            rg_mapper = ResourceGridMapper(rg)

            num_data_bits = int(rg.num_data_symbols * link_config.num_bits_per_symbol)
            b = bs([batch_size, link_config.num_tx_ant, 1, num_data_bits])
            if device is not None and str(b.device) != str(device):
                b = b.to(device)

            x = mapper(b)
            x_rg = rg_mapper(x)

            no_tensor = torch.tensor([float(ebno_db)], device=x_rg.device)
            no_val = ebnodb2no(no_tensor, link_config.num_bits_per_symbol, link_config.coderate)
            no_ = expand_to_rank(no_val, x_rg.ndim)

            ofdm_channel = OFDMChannel(channel, rg, normalize_channel=True, return_channel=True)
            y, h = ofdm_channel(x_rg, no_)

            # Squeeze BS receiver dimension: [batch, 1, rx_ant, symbols, fft] -> [batch, rx_ant, symbols, fft]
            Y = y.squeeze(1) if y.ndim == 5 else y

            # Shape H to [batch, rx_ant, tx, symbols, fft]
            if h.ndim == 7:
                H = h[:, 0, :, 0, :, :, :]
            elif h.ndim == 6:
                H = h.squeeze(1)
            else:
                H = h

            X_pilot = rg.pilot_pattern.pilots
            N0 = float(no_val.item())

            return BatchData(b=b, Y=Y, X_pilot=X_pilot, N0=N0, H=H)

        else:
            # Pure PyTorch synthetic fallback
            num_data_bits = link_config.num_data_bits
            b = torch.randint(
                0, 2, (batch_size, link_config.num_tx_ant, 1, num_data_bits), device=device, dtype=torch.float32
            )
            N0 = float(10.0 ** (-float(ebno_db) / 10.0) / (link_config.num_bits_per_symbol * link_config.coderate))

            # Channel response
            if hasattr(channel, "sample"):
                H = channel.sample(
                    batch_size=batch_size,
                    num_ofdm_symbols=link_config.num_ofdm_symbols,
                    fft_size=link_config.fft_size,
                )
            elif callable(channel):
                H = channel(batch_size)
            else:
                h_r = torch.randn(
                    (batch_size, link_config.num_rx_ant, link_config.num_tx_ant, link_config.num_ofdm_symbols, link_config.fft_size),
                    device=device,
                )
                h_i = torch.randn(
                    (batch_size, link_config.num_rx_ant, link_config.num_tx_ant, link_config.num_ofdm_symbols, link_config.fft_size),
                    device=device,
                )
                H = torch.complex(h_r, h_i) / math.sqrt(2.0)

            if H.dim() == 7:
                H = H[:, 0, :, 0, :, :, :]

            # Signal + noise
            from src.rx.classical import _get_qam_cache, _get_active_subcarrier_indices

            bps = link_config.num_bits_per_symbol
            pts, _, _ = _get_qam_cache(bps, device)
            b_grouped = b.view(batch_size, -1, bps).long()
            powers = 2 ** torch.arange(bps - 1, -1, -1, device=device).long()
            indices = torch.sum(b_grouped * powers, dim=-1)
            s_data = pts[indices]

            active_sc = _get_active_subcarrier_indices(link_config)
            pilot_syms = set(link_config.pilot_ofdm_symbol_indices)
            data_syms = [t for t in range(link_config.num_ofdm_symbols) if t not in pilot_syms]

            X_grid = torch.zeros(
                (batch_size, 1, link_config.num_ofdm_symbols, link_config.fft_size),
                dtype=torch.complex64,
                device=device,
            )
            for p_idx in pilot_syms:
                X_grid[:, 0, p_idx, active_sc] = 1.0 + 0j

            s_data_reshaped = s_data.view(batch_size, len(data_syms), len(active_sc))
            for i, d_idx in enumerate(data_syms):
                X_grid[:, 0, d_idx, active_sc] = s_data_reshaped[:, i, :]

            noise_r = torch.randn(
                (batch_size, link_config.num_rx_ant, link_config.num_ofdm_symbols, link_config.fft_size),
                device=device,
            ) * math.sqrt(N0 / 2.0)
            noise_i = torch.randn(
                (batch_size, link_config.num_rx_ant, link_config.num_ofdm_symbols, link_config.fft_size),
                device=device,
            ) * math.sqrt(N0 / 2.0)
            noise = torch.complex(noise_r, noise_i)

            Y = H[:, :, 0, :, :] * X_grid + noise
            X_pilot = torch.ones((1, 1, link_config.num_pilot_symbols), dtype=torch.complex64, device=device)

            return BatchData(b=b, Y=Y, X_pilot=X_pilot, N0=N0, H=H)

    if as_generator:
        def _generator_wrapper() -> Generator[BatchData, None, None]:
            while True:
                yield _generate_single_batch()

        return _generator_wrapper()

    return _generate_single_batch()
