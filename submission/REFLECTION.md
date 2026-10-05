# Reflection — Lab 19

**Tên:** Đinh Đức Thái
**Cohort:** A20-K4
**Path đã chạy:** lite (Python 3.11, fastembed, Qdrant in-memory, SQLite Feast)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact**: BM25 thắng/ngang Hybrid (96.7% vs 88.7% Vector) vì chứa thuật ngữ kỹ thuật verbatim (Kubernetes, OAuth, VPC) mà inverted index bắt trúng 100%.
- **Paraphrase**: Cả hai giảm điểm do model `bge-small-en` không tối ưu tiếng Việt (chuyển sang `bge-m3` ở Docker path sẽ vượt trội). Vector bắt ý niệm tốt hơn keyword thuần.
- **Mixed**: **Hybrid thắng tuyệt đối (100.0%)** so với BM25 (97.0%) và Vector (98.5%) nhờ RRF ($k=60$) cộng hưởng tín hiệu từ vựng chính xác và ngữ nghĩa mở rộng.

**Khi nào KHÔNG dùng Hybrid:**
1. **Dùng Pure BM25** khi query là thực thể định danh chính xác (mã SKU, UUID, lỗi log, tên hàm API) hoặc hệ thống yêu cầu độ trễ cực thấp (< 2ms) mà không tốn chi phí inference vector.
2. **Dùng Pure Vector** cho tìm kiếm đa ngôn ngữ (cross-lingual), đa phương thức (multimodal), hoặc trừu tượng không chứa từ khóa trùng khớp.

---

## Điều ngạc nhiên nhất khi làm lab này

Hiệu ứng "Recall Cliff" ở NB5 khi post-filter sập về 0% ở mức chọn lọc 4% khiến hệ thống hỏng âm thầm mà không văng lỗi; và việc thiếu namespace ở Semantic Cache (NB7) dẫn đến rò rỉ dữ liệu chéo tenant tức thì.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _tự thực hiện_
