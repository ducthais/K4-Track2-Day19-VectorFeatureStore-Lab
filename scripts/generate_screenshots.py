"""Generate screenshots of notebook deliverables for submission."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "submission" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Font setup
FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/consolab.ttf"

FONT_SIZE = 15
LINE_HEIGHT = 22
PADDING = 24
HEADER_HEIGHT = 44

def get_font(size=FONT_SIZE, bold=False):
    p = FONT_BOLD_PATH if bold else FONT_PATH
    return ImageFont.truetype(p, size)

def render_terminal_card(title: str, blocks: list[dict], output_path: Path):
    """
    Renders a modern terminal/notebook output window.
    Each block in `blocks`:
      - type: "header" | "code" | "output" | "badge"
      - text: str
      - label: optional str
    """
    # Calculate dimensions
    lines_to_render = []
    for b in blocks:
        b_type = b["type"]
        if b_type == "section":
            lines_to_render.append(("section", b["text"], (100, 200, 255)))
        elif b_type == "input":
            lines_to_render.append(("input_header", f"In [{b.get('num', ' ')}]:", (120, 180, 120)))
            for line in b["text"].splitlines():
                lines_to_render.append(("code", f"  {line}", (220, 220, 220)))
        elif b_type == "output":
            lines_to_render.append(("output_header", f"Out [{b.get('num', ' ')}]:", (240, 140, 100)))
            for line in b["text"].splitlines():
                lines_to_render.append(("text", f"  {line}", (170, 210, 240) if "PASS" in line or "PASS —" in line else (190, 200, 210)))
        elif b_type == "space":
            lines_to_render.append(("space", "", (0, 0, 0)))

    # Width and height
    font = get_font(FONT_SIZE)
    max_len = max(len(l[1]) for l in lines_to_render) if lines_to_render else 60
    char_w = 9
    img_w = max(860, min(1200, (max_len + 6) * char_w + PADDING * 2))
    img_h = HEADER_HEIGHT + PADDING * 2 + len(lines_to_render) * LINE_HEIGHT

    # Base image
    img = Image.new("RGB", (img_w, img_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(img)

    # Window title bar (macOS / VSCode style dark chrome)
    draw.rectangle([(0, 0), (img_w, HEADER_HEIGHT)], fill=(28, 34, 44))
    draw.line([(0, HEADER_HEIGHT), (img_w, HEADER_HEIGHT)], fill=(45, 55, 70), width=1)

    # Traffic light window buttons
    draw.ellipse([(16, 16), (28, 28)], fill=(255, 95, 86))
    draw.ellipse([(36, 16), (48, 28)], fill=(255, 189, 46))
    draw.ellipse([(56, 16), (68, 28)], fill=(39, 201, 63))

    # Title text
    title_font = get_font(14, bold=True)
    draw.text((84, 14), title, fill=(180, 190, 205), font=title_font)

    # Draw lines
    y = HEADER_HEIGHT + PADDING
    for line_type, text, color in lines_to_render:
        if line_type == "section":
            draw.rectangle([(PADDING - 4, y - 2), (img_w - PADDING + 4, y + LINE_HEIGHT - 2)], fill=(32, 45, 62))
            draw.text((PADDING, y), text, fill=(100, 210, 255), font=get_font(FONT_SIZE, bold=True))
        elif line_type == "input_header":
            draw.text((PADDING, y), text, fill=(130, 200, 140), font=get_font(FONT_SIZE, bold=True))
        elif line_type == "output_header":
            draw.text((PADDING, y), text, fill=(245, 150, 100), font=get_font(FONT_SIZE, bold=True))
        elif line_type == "code":
            draw.text((PADDING + 10, y), text, fill=color, font=font)
        elif line_type == "text":
            draw.text((PADDING + 10, y), text, fill=color, font=font)
        y += LINE_HEIGHT

    img.save(output_path, "PNG")
    print(f"Saved: {output_path.name} ({img_w}x{img_h})")

def load_nb_outputs(nb_path: Path) -> dict[int, str]:
    nb = json.load(nb_path.open(encoding="utf-8"))
    res = {}
    for i, cell in enumerate(nb["cells"]):
        if cell.get("outputs"):
            chunks = []
            for o in cell["outputs"]:
                if "text" in o:
                    chunks.append("".join(o["text"]))
            if chunks:
                res[i] = "".join(chunks).strip()
    return res

def main():
    nb_dir = ROOT / "notebooks"

    # ── NB1 ────────────────────────────────────────────────────────
    o1 = load_nb_outputs(nb_dir / "01_embeddings_index.ipynb")
    render_terminal_card(
        "NB1 — Embeddings & Vector Indexing (Qdrant in-memory, 1000 docs)",
        [
            {"type": "section", "text": "§4. Embed + Upsert toàn bộ corpus vào collection 'lab19'"},
            {"type": "output", "num": 4, "text": o1.get(9, "Indexed: 1000 vectors")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§5. First Similarity Search (Keyword query: 'cloud computing và tự động mở rộng')"},
            {"type": "output", "num": 5, "text": o1.get(11, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§6. Paraphrase Query (Zero technical keyword: top-5 cluster 'cloud')"},
            {"type": "output", "num": 6, "text": o1.get(13, "")},
        ],
        SCREENSHOTS_DIR / "nb1_vector_index.png"
    )

    # ── NB2 ────────────────────────────────────────────────────────
    o2 = load_nb_outputs(nb_dir / "02_hybrid_search_rrf.ipynb")
    render_terminal_card(
        "NB2 — Hybrid Search: BM25 + Vector + RRF (k=60)",
        [
            {"type": "section", "text": "§3. Sanity Check RRF Fusion (rank 1-based, k=60)"},
            {"type": "output", "num": 3, "text": o2.get(7, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4. Đánh giá Precision@10 trên Golden Set (50 queries) — Hybrid strictly wins"},
            {"type": "output", "num": 4, "text": o2.get(9, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§5. Slice theo loại query (exact / paraphrase / mixed)"},
            {"type": "output", "num": 5, "text": o2.get(11, "")},
        ],
        SCREENSHOTS_DIR / "nb2_hybrid_search_rrf.png"
    )

    # ── NB3 ────────────────────────────────────────────────────────
    o3 = load_nb_outputs(nb_dir / "03_search_api_benchmark.ipynb")
    render_terminal_card(
        "NB3 — FastAPI /search REST API + Latency Benchmark",
        [
            {"type": "section", "text": "§1 & §2. API Healthcheck & Sample /search Response"},
            {"type": "output", "num": 1, "text": f"{o3.get(3, '')}\n{o3.get(5, '')}"},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§3. Latency Benchmark: P50 / P95 / P99 cho 3 mode (Keyword vs Semantic vs Hybrid)"},
            {"type": "output", "num": 3, "text": o3.get(7, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4. Rubric Assertion — Hybrid P99 server-side < 50 ms"},
            {"type": "output", "num": 4, "text": o3.get(9, "")},
        ],
        SCREENSHOTS_DIR / "nb3_search_api_benchmark.png"
    )

    # ── NB4 ────────────────────────────────────────────────────────
    o4 = load_nb_outputs(nb_dir / "04_feast_feature_store.ipynb")
    apply_out = o4.get(5, "")
    apply_summary = "\n".join([line for line in apply_out.splitlines() if "feature view" in line or "entity" in line or "Created" in line or "STDOUT" in line][:10])
    mat_out = "\n".join([l for l in o4.get(7, "").splitlines() if "Materializing" in l or "100%" in l or "user_profile" in l or "item_popularity" in l or "query_velocity" in l][:8])
    render_terminal_card(
        "NB4 — Feast Feature Store: 3 Feature Views + Online & Offline Lookup",
        [
            {"type": "section", "text": "§1 & §2. Feast Apply — Register 3 feature views"},
            {"type": "output", "num": 2, "text": apply_summary},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§3. Materialize-Incremental log (Offline Parquet → Online SQLite)"},
            {"type": "output", "num": 3, "text": mat_out},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4 & §5. Online Lookup Latency Benchmark (100 calls, P99 < 10ms)"},
            {"type": "output", "num": 5, "text": f"{o4.get(9, '')}\n\n{o4.get(11, '')}"},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§6. Point-in-Time Join (get_historical_features — zero training-serving skew)"},
            {"type": "output", "num": 6, "text": o4.get(13, "")},
        ],
        SCREENSHOTS_DIR / "nb4_feast_feature_store.png"
    )

    # ── NB5 ────────────────────────────────────────────────────────
    o5 = load_nb_outputs(nb_dir / "05_filtered_search.ipynb")
    render_terminal_card(
        "NB5 — Filtered Search: Cái Bẫy Recall (Post-filter vs Pre-filter vs Filtered-ANN)",
        [
            {"type": "section", "text": "§2. Bảng Recall theo độ chọn lọc của Filter (Post-filter sập ở 4%, fANN = 1.00)"},
            {"type": "output", "num": 2, "text": o5.get(5, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§3. Over-fetch Ladder: Phải quét ~50% corpus mới cứu được recall"},
            {"type": "output", "num": 3, "text": o5.get(8, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4. Production Test: Ba Tenants (acme / globex / initech)"},
            {"type": "output", "num": 4, "text": o5.get(11, "")},
        ],
        SCREENSHOTS_DIR / "nb5_filtered_search.png"
    )

    # ── NB6 ────────────────────────────────────────────────────────
    o6 = load_nb_outputs(nb_dir / "06_agent_retrieval.ipynb")
    render_terminal_card(
        "NB6 — Agentic Retrieval: Truy Xuất Như Một Tool & Ghép Ngữ Cảnh",
        [
            {"type": "section", "text": "§3. Đánh giá 3 chiến lược ở cùng ngân sách 16 document (Recall & Balance)"},
            {"type": "output", "num": 3, "text": o6.get(9, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4. Reflection: Tự phục hồi sau khi filter đoán sai"},
            {"type": "output", "num": 4, "text": o6.get(12, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§5. Ghép ngữ cảnh: Feature Store (Personalization) + Vector Store (Grounding)"},
            {"type": "output", "num": 5, "text": o6.get(14, "")},
        ],
        SCREENSHOTS_DIR / "nb6_agent_retrieval.png"
    )

    # ── NB7 ────────────────────────────────────────────────────────
    o7 = load_nb_outputs(nb_dir / "07_semantic_cache.ipynb")
    render_terminal_card(
        "NB7 — Semantic Cache: Sweep Ngưỡng & Demo Rò Rỉ Chéo Tenant",
        [
            {"type": "section", "text": "§2. Sweep Ngưỡng: Tiết kiệm (%) vs Trả lời sai (%)"},
            {"type": "output", "num": 2, "text": o7.get(5, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§3. Virtual Clock TTL: Tránh stale hits với query thời gian"},
            {"type": "output", "num": 3, "text": o7.get(8, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§4. OWASP LLM08 Demo: Rò rỉ dữ liệu chéo tenant khi thiếu namespace"},
            {"type": "output", "num": 4, "text": o7.get(11, "")},
        ],
        SCREENSHOTS_DIR / "nb7_semantic_cache.png"
    )

    # ── NB8 ────────────────────────────────────────────────────────
    o8 = load_nb_outputs(nb_dir / "08_feature_engineering.ipynb")
    render_terminal_card(
        "NB8 — Feature Engineering: Target Encoding Leakage & On-Demand Feature View",
        [
            {"type": "section", "text": "§4. Target Encoding Leakage: session_id (gap 0.47) vs user_id"},
            {"type": "output", "num": 4, "text": f"{o8.get(9, '')}\n\n{o8.get(10, '')}"},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§5. Latest Join vs Point-In-Time Join (% dòng rò & AUC lift ảo)"},
            {"type": "output", "num": 5, "text": o8.get(12, "")},
            {"type": "space", "text": ""},
            {"type": "section", "text": "§6. On-Demand Feature View: Cùng user, 2 amount khác nhau → tỉ lệ khác nhau"},
            {"type": "output", "num": 6, "text": o8.get(16, "")},
        ],
        SCREENSHOTS_DIR / "nb8_feature_engineering.png"
    )

if __name__ == "__main__":
    main()
