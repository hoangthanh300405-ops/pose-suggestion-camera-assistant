# Development notes

Problems met while building this, kept so they are not repeated.

- **Template/keypoint names must match.** A mock template used `left_shoulder` while the matcher used
  MPII names (`l_shoulder`), leaving 2 joints in common and a score of 0.0. v2 canonicalises names and
  warns when a template has fewer than 10 joints or an id starting with `demo_`.
- **Normalise in pixels.** MediaPipe returns 0-1 coordinates; normalising them directly distorts the
  skeleton on non-square images while MPII templates are in isotropic pixels.
- **OpenCV `putText` cannot draw Vietnamese.** Captions are drawn with PIL and a Unicode font.
- **MPII activity keywords need word stems.** The dataset says "dancing", so the keyword "dance" matches nothing.
- **Colab:** do not paste scripts into a cell that parses `sys.argv` (Colab adds `-f kernel.json`); keep code
  in files on Drive and import functions. Never let an automatic "fix" tool overwrite data files.
- **Core ML:** conversion works on Colab, but prediction (and the parity check) needs macOS. coremltools
  officially tests up to torch 2.7, so check parity before trusting an export made with newer torch.
