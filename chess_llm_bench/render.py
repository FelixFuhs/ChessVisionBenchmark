from pathlib import Path
from typing import Optional, List

import os
import platform
import chess
from PIL import Image, ImageDraw, ImageFont


LIGHT = (240, 217, 181)  # #F0D9B5
DARK = (181, 136, 99)    # #B58863
TEXT = (20, 20, 20)
COORD = (30, 30, 30)

# Unicode chess glyphs
CHESS_GLYPHS = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "q": "\u265B", "r": "\u265C", "b": "\u265D", "n": "\u265E", "p": "\u265F",
}


def _font_paths_candidates() -> List[str]:
    # Try common fonts that include Unicode chess glyphs
    candidates = []
    system = platform.system()
    if system == "Windows":
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts")
        candidates += [
            os.path.join(win_fonts, "seguisym.ttf"),  # Segoe UI Symbol
            os.path.join(win_fonts, "seguiemj.ttf"),  # Segoe UI Emoji
            os.path.join(win_fonts, "arialuni.ttf"),  # Arial Unicode MS (if present)
            os.path.join(win_fonts, "DejaVuSans.ttf"),
        ]
    elif system == "Darwin":
        candidates += [
            "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
            "/Library/Fonts/NotoSansSymbols2-Regular.ttf",
        ]
    else:  # Linux
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/truetype/Symbola/Symbola.ttf",
        ]
    return candidates


def _load_chess_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # 1) Explicit override via env var
    p = os.environ.get("CHESS_FONT_PATH")
    if p and os.path.exists(p):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    # 2) Search common candidates
    for p in _font_paths_candidates():
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    # Fallback to default font (may not have chess glyphs)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_board_png(
    board: chess.Board,
    out_path: Path,
    size_px: int = 512,
    coordinates: bool = True,
    lastmove: Optional[chess.Move] = None,
) -> None:
    """Render a board to a PNG file at the given size using Pillow only.

    - White at bottom (rank 1).
    - Uppercase pieces for White, lowercase for Black.
    - Optionally draws simple coordinates inside edge squares.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Board grid without external margins (512x512 for 8x8 tiles -> 64px each)
    img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    tile = size_px // 8
    # Fallback fonts; default is small but sufficient. Try a slightly larger if available.
    font = _load_chess_font(tile // 2)
    try:
        coord_font = ImageFont.truetype("arial.ttf", max(10, tile // 6))
    except Exception:
        coord_font = ImageFont.load_default()

    # Highlight last move squares if provided
    last_from = last_to = None
    if lastmove is not None:
        last_from = lastmove.from_square
        last_to = lastmove.to_square

    for rank in range(7, -1, -1):  # 7..0 top to bottom
        for file in range(8):       # 0..7 left to right
            sq = chess.square(file, rank)
            x0 = file * tile
            y0 = (7 - rank) * tile
            x1 = x0 + tile
            y1 = y0 + tile

            color = LIGHT if (file + rank) % 2 == 0 else DARK
            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Simple last move highlight overlay (slight tint)
            if last_from == sq or last_to == sq:
                draw.rectangle([x0, y0, x1, y1], outline=(255, 215, 0), width=3)

            piece = board.piece_at(sq)
            if piece:
                s = piece.symbol()  # uppercase white, lowercase black
                glyph = CHESS_GLYPHS.get(s, s)
                # Center the text
                bbox = draw.textbbox((0, 0), glyph, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                cx = x0 + (tile - w) / 2
                cy = y0 + (tile - h) / 2
                draw.text((cx, cy), glyph, fill=TEXT, font=font)

            # Coordinates inside edge squares
            if coordinates:
                if rank == 0:  # bottom files (a..h)
                    file_letter = chr(ord('a') + file)
                    draw.text((x0 + 4, y1 - 14), file_letter, fill=COORD, font=coord_font)
                if file == 0:  # left ranks (1..8)
                    rank_digit = str(rank + 1)
                    draw.text((x0 + 4, y0 + 2), rank_digit, fill=COORD, font=coord_font)

    img.save(out_path, format="PNG")
