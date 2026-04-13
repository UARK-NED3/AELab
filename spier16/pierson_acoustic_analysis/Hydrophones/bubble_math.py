import math


def bubble_properties(flow_rate_ul_per_min, seconds_per_bubble):
    """
    Compute bubble volume and equivalent spherical radius.

    Parameters
    ----------
    flow_rate_ul_per_min : float
        Flow rate from syringe pump (µL/min).
    seconds_per_bubble : float
        Time between bubbles (seconds).

    Returns
    -------
    volume_ul : float
        Volume of a bubble in µL.
    radius_mm : float
        Radius of an equivalent spherical bubble in mm.
    """

    # Convert flow rate to µL/s
    flow_rate_ul_per_s = flow_rate_ul_per_min / 60.0

    # Volume per bubble
    volume_ul = flow_rate_ul_per_s * seconds_per_bubble

    # 1 µL = 1 mm^3
    volume_mm3 = volume_ul

    # Radius of equivalent sphere
    radius_mm = ((3 * volume_mm3) / (4 * math.pi)) ** (1 / 3)

    return volume_ul, radius_mm


def theoretical_radius_from_minnaert_frequency(
    frequency_hz,
    hydrostatic_pressure_in_h2o,
    gamma=1.4,
    water_density_kg_per_m3=997.0,
    atmospheric_pressure_pa=101325.0,
):
    """
    Compute the theoretical bubble radius from the Minnaert frequency.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")
    if water_density_kg_per_m3 <= 0:
        raise ValueError("water_density_kg_per_m3 must be positive.")

    # Conversion: 1 inch of water ≈ 249.0889 Pa
    inches_h2o_to_pa = 249.0889
    hydrostatic_pressure_pa = hydrostatic_pressure_in_h2o * inches_h2o_to_pa

    ambient_pressure_pa = atmospheric_pressure_pa + hydrostatic_pressure_pa

    radius_m = (1.0 / (2.0 * math.pi * frequency_hz)) * math.sqrt(
        (3.0 * gamma * ambient_pressure_pa) / water_density_kg_per_m3
    )

    radius_mm = radius_m * 1000.0
    return radius_mm, ambient_pressure_pa


def theoretical_minnaert_frequency_from_radius(
    radius_mm,
    hydrostatic_pressure_in_h2o,
    gamma=1.4,
    water_density_kg_per_m3=997.0,
    atmospheric_pressure_pa=101325.0,
):
    """
    Compute the theoretical Minnaert frequency from bubble radius.

    Parameters
    ----------
    radius_mm : float
        Bubble radius in mm.
    hydrostatic_pressure_in_h2o : float
        Hydrostatic pressure at the bubble location, in inches of water.
    gamma : float, optional
        Polytropic exponent for the gas. Default is 1.4.
    water_density_kg_per_m3 : float, optional
        Density of water in kg/m^3. Default is 997.0.
    atmospheric_pressure_pa : float, optional
        Atmospheric pressure in Pa. Default is 101325.0.

    Returns
    -------
    frequency_hz : float
        Predicted Minnaert frequency in Hz.
    ambient_pressure_pa : float
        Total ambient pressure used in the calculation, in Pa.
    """
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive.")
    if water_density_kg_per_m3 <= 0:
        raise ValueError("water_density_kg_per_m3 must be positive.")

    # Convert pressure
    inches_h2o_to_pa = 249.0889
    hydrostatic_pressure_pa = hydrostatic_pressure_in_h2o * inches_h2o_to_pa
    ambient_pressure_pa = atmospheric_pressure_pa + hydrostatic_pressure_pa

    # Convert mm to m
    radius_m = radius_mm / 1000.0

    frequency_hz = (1.0 / (2.0 * math.pi * radius_m)) * math.sqrt(
        (3.0 * gamma * ambient_pressure_pa) / water_density_kg_per_m3
    )

    return frequency_hz, ambient_pressure_pa


if __name__ == "__main__":
    # Example values for flow/volume-based radius
    flow_rate = 180.5 / 2       # µL/min
    bubble_interval = 5.8302    # s

    volume, radius = bubble_properties(flow_rate, bubble_interval)

    print(f"Flow rate: {flow_rate:.4f} µL/min")
    print(f"Bubble interval: {bubble_interval:.4f} s")
    print()
    print(f"Bubble volume: {volume:.4f} µL")
    print(f"Equivalent spherical radius: {radius:.4f} mm")
    print(f"Equivalent spherical diameter: {radius * 2:.4f} mm")

    print("\n" + "-" * 50)

    # Use the flow/interval-derived radius to predict Minnaert frequency
    hydrostatic_head_in_h2o = 6.0  # inches of water

    predicted_frequency_hz, ambient_pressure_pa = theoretical_minnaert_frequency_from_radius(
        radius,
        hydrostatic_head_in_h2o,
    )

    print("Predicted Minnaert frequency from flow-rate-derived radius")
    print(f"Hydrostatic pressure: {hydrostatic_head_in_h2o:.2f} inH2O")
    print(f"Ambient pressure used: {ambient_pressure_pa:.2f} Pa")
    print(f"Predicted Minnaert frequency: {predicted_frequency_hz:.2f} Hz")

    print("\n" + "-" * 50)

    # Example values for frequency-based theoretical radius
    minnaert_frequency = 2450.0  # Hz

    theoretical_radius_mm, ambient_pressure_pa = theoretical_radius_from_minnaert_frequency(
        minnaert_frequency,
        hydrostatic_head_in_h2o,
    )

    print(f"Measured Minnaert frequency: {minnaert_frequency:.2f} Hz")
    print(f"Hydrostatic pressure: {hydrostatic_head_in_h2o:.2f} inH2O")
    print(f"Ambient pressure used: {ambient_pressure_pa:.2f} Pa")
    print(f"Theoretical Minnaert radius: {theoretical_radius_mm:.4f} mm")
    print(f"Theoretical Minnaert diameter: {2 * theoretical_radius_mm:.4f} mm")