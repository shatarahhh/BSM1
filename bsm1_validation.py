def main():
    csv_path = find_csv()
    out_dir = os.path.dirname(csv_path)
    set_style()
    rows, ref_vals, our_vals, pct_diff = load_summary(csv_path)

    # Safety: ensure manual labels length matches CSV rows
    if len(SHORT_TICK_LABELS) != len(rows):
        raise ValueError(f"SHORT_TICK_LABELS has {len(SHORT_TICK_LABELS)} items, "
                         f"but CSV has {len(rows)} rows. Please adjust the list.")

    # NEW: append units from CSV under each short label (second line)
    units = [r["unit"] for r in rows]
    final_labels = [f"{short}\n[{unit}]" for short, unit in zip(SHORT_TICK_LABELS, units)]

    make_plot(final_labels, ref_vals, our_vals, pct_diff, out_dir)
