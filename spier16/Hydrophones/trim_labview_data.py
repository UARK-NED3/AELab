from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim trailing LabVIEW data rows from a .lvm or .txt file when the "
            "first column exceeds a threshold."
        )
    )
    parser.add_argument("input_path", type=Path, help="Path to the input .lvm or .txt file")
    parser.add_argument(
        "threshold",
        type=float,
        help="Trim trailing rows while the first column is greater than this value",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        help="Optional path for the trimmed output file",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new file",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, output_path: Path | None, in_place: bool) -> Path:
    if output_path and in_place:
        raise ValueError("Use either --output or --in-place, not both.")

    if output_path is not None:
        return output_path

    if in_place:
        return input_path

    return input_path.with_name(f"{input_path.stem}_trimmed{input_path.suffix}")


def trim_labview_lines(lines: list[str], threshold: float) -> tuple[list[str], int]:
    data_header_index = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("X_Value,")),
        -1,
    )
    if data_header_index < 0:
        raise ValueError("Could not find the data header line starting with 'X_Value,'.")

    kept_lines = lines[: data_header_index + 1]
    data_lines = lines[data_header_index + 1 :]

    last_line_to_keep = len(data_lines) - 1
    while last_line_to_keep >= 0:
        candidate = data_lines[last_line_to_keep].strip()

        if not candidate:
            last_line_to_keep -= 1
            continue

        first_field = candidate.split(",", 1)[0].strip()
        try:
            parsed_value = float(first_field)
        except ValueError:
            break

        if parsed_value > threshold:
            last_line_to_keep -= 1
            continue

        break

    kept_lines.extend(data_lines[: last_line_to_keep + 1])
    removed_count = len(data_lines) - (last_line_to_keep + 1)
    return kept_lines, removed_count


def main() -> None:
    args = parse_args()

    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = build_output_path(input_path, args.output_path, args.in_place)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    trimmed_lines, removed_count = trim_labview_lines(lines, args.threshold)
    output_path.write_text("\n".join(trimmed_lines) + "\n", encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output: {output_path.resolve()}")
    print(f"Threshold: {args.threshold}")
    print(f"Removed trailing rows: {removed_count}")


if __name__ == "__main__":
    main()
