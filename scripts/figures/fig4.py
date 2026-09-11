r"""Fig. 4 -- Photographic challenge space. Eight full I* images, 2x4,
no ROI markers, no inset -- just the eight frozen ideal all-in-focus
references at full size."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from common import save, panel_label, imshow_gray, load_npy

CONDS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE", "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]


def main():
    fig = plt.figure(figsize=(11, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    letters = "abcdefgh"

    for i, cond in enumerate(CONDS):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        full = load_npy(f"01_photographic_conditions/Istar_{cond}_full.npy")
        imshow_gray(ax, full, title=cond, title_fontsize=12, title_pad=10)
        panel_label(ax, letters[i], x=-0.03, y=1.20)

    save(fig, "Figure4", dpi=1050)


if __name__ == "__main__":
    main()
