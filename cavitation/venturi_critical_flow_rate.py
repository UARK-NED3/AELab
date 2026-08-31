from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class CavitationResult:
    flow_m3_s: float
    flow_l_min: float
    inlet_velocity_m_s: float
    throat_velocity_m_s: float
    vapor_pressure_pa: float


def water_vapor_pressure_pa(temperature_c: float) -> float:
    """
    Estimate the saturation vapor pressure of water using the Antoine equation.

    Valid approximately from 1°C to 100°C.

    Returns
    -------
    float
        Vapor pressure in Pa.
    """
    if not 1.0 <= temperature_c <= 100.0:
        raise ValueError(
            "This Antoine correlation is intended for water from 1°C to 100°C."
        )

    # Antoine constants for water.
    # Returns pressure in mmHg when temperature is in °C.
    A = 8.07131
    B = 1730.63
    C = 233.426

    vapor_pressure_mmhg = 10 ** (A - B / (C + temperature_c))
    return vapor_pressure_mmhg * 133.322368


def critical_venturi_flow(
    inlet_diameter_m: float,
    throat_diameter_m: float,
    inlet_absolute_pressure_pa: float,
    water_temperature_c: float,
    water_density_kg_m3: float = 998.2,
    inlet_minus_throat_elevation_m: float = 0.0,
    inlet_to_throat_loss_coefficient: float = 0.0,
) -> CavitationResult:
    """
    Calculate the ideal critical flow rate for cavitation inception.

    Cavitation inception is estimated by setting:

        throat absolute pressure = water vapor pressure

    The calculation applies Bernoulli's equation between the Venturi inlet
    and throat, with an optional loss coefficient referenced to throat velocity.

    Parameters
    ----------
    inlet_diameter_m:
        Internal diameter at the Venturi inlet, in meters.

    throat_diameter_m:
        Internal throat diameter, in meters.

    inlet_absolute_pressure_pa:
        Absolute static pressure immediately upstream of the Venturi, in Pa.
        Do not supply gauge pressure directly.

    water_temperature_c:
        Water temperature in °C.

    water_density_kg_m3:
        Water density in kg/m³.

    inlet_minus_throat_elevation_m:
        z_inlet - z_throat, in meters.
        Use 0 for a horizontal Venturi.

    inlet_to_throat_loss_coefficient:
        Loss coefficient K between the inlet pressure tap and throat,
        referenced to throat velocity. Use 0 for the ideal estimate.

    Returns
    -------
    CavitationResult
        Critical flow rate and associated velocities.
    """
    if inlet_diameter_m <= 0 or throat_diameter_m <= 0:
        raise ValueError("Diameters must be positive.")

    if throat_diameter_m >= inlet_diameter_m:
        raise ValueError(
            "The throat diameter must be smaller than the inlet diameter."
        )

    if inlet_absolute_pressure_pa <= 0:
        raise ValueError("Absolute inlet pressure must be positive.")

    if water_density_kg_m3 <= 0:
        raise ValueError("Water density must be positive.")

    if inlet_to_throat_loss_coefficient < 0:
        raise ValueError("Loss coefficient cannot be negative.")

    inlet_area_m2 = pi * inlet_diameter_m**2 / 4
    throat_area_m2 = pi * throat_diameter_m**2 / 4

    vapor_pressure_pa = water_vapor_pressure_pa(water_temperature_c)

    gravity_m_s2 = 9.80665

    available_pressure_pa = (
        inlet_absolute_pressure_pa
        - vapor_pressure_pa
        + water_density_kg_m3
        * gravity_m_s2
        * inlet_minus_throat_elevation_m
    )

    if available_pressure_pa <= 0:
        raise ValueError(
            "The inlet pressure is already at or below the predicted "
            "cavitation threshold after accounting for elevation."
        )

    denominator = water_density_kg_m3 * (
        (1 + inlet_to_throat_loss_coefficient) / throat_area_m2**2
        - 1 / inlet_area_m2**2
    )

    if denominator <= 0:
        raise ValueError(
            "The selected geometry and loss coefficient produce an "
            "invalid denominator."
        )

    flow_m3_s = sqrt(2 * available_pressure_pa / denominator)

    return CavitationResult(
        flow_m3_s=flow_m3_s,
        flow_l_min=flow_m3_s * 60_000,
        inlet_velocity_m_s=flow_m3_s / inlet_area_m2,
        throat_velocity_m_s=flow_m3_s / throat_area_m2,
        vapor_pressure_pa=vapor_pressure_pa,
    )


if __name__ == "__main__":
    # Example:
    # 12 mm inlet, 6 mm throat, 150 kPa absolute inlet pressure,
    # horizontal Venturi, water at 20°C.
    result = critical_venturi_flow(
        inlet_diameter_m=12e-3,
        throat_diameter_m=6e-3,
        inlet_absolute_pressure_pa=150e3,
        water_temperature_c=20.0,
        inlet_minus_throat_elevation_m=0.0,
        inlet_to_throat_loss_coefficient=0.0,
    )

    print(f"Water vapor pressure: {result.vapor_pressure_pa / 1000:.3f} kPa abs")
    print(f"Critical flow rate:   {result.flow_m3_s:.6g} m³/s")
    print(f"Critical flow rate:   {result.flow_l_min:.2f} L/min")
    print(f"Inlet velocity:       {result.inlet_velocity_m_s:.2f} m/s")
    print(f"Throat velocity:      {result.throat_velocity_m_s:.2f} m/s")