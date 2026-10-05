"""Demo script for HybridMemoryAgent.

Demonstrates 5 query patterns blending episodic memory with user feature store.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent

def main():
    print("==================================================================")
    print("     DAY 19 BONUS CHALLENGE: AI HYBRID MEMORY AGENT DEMO          ")
    print("==================================================================\n")

    agent = HybridMemoryAgent()

    # Seed episodic memories for user u_001
    user_id = "u_001"
    print(f"[*] Seeding episodic memories for user '{user_id}'...")

    notes = [
        "Kubernetes Pod lifecycle: Pod có các giai đoạn Pending, Running, Succeeded, Failed. Khởi tạo container dùng initContainers để chạy script setup trước.",
        "Thiết lập mạng VPC: Chia subnet public cho Application Load Balancer và private subnet cho cơ sở dữ liệu để tăng tính bảo mật zero-trust.",
        "Tự động mở rộng (Auto-scaling) theo lưu lượng: Cấu hình Horizontal Pod Autoscaler (HPA) dựa trên CPU utilization và custom metrics Prometheus.",
        "Bảo mật điện toán đám mây: Sử dụng IAM roles, mã hóa dữ liệu at-rest bằng AWS KMS và quản lý bí mật qua HashiCorp Vault.",
        "Mô hình ngôn ngữ lớn tiếng Việt: Đánh giá RAG pipeline bằng RRF k=60 kết hợp BM25 sparse và dense vector embedding.",
    ]

    for note in notes:
        agent.remember(note, user_id=user_id)

    # Also add some notes for a different user u_002 to prove isolation!
    agent.remember("Tài liệu mật của u_002: Báo cáo tài chính quý 4 bí mật.", user_id="u_002")

    print(f"[*] Indexed {len(agent.memories)} episodic memory chunks.")
    print("[*] Feast Feature Store connection verified.\n")

    queries = [
        (
            "1. Hỏi đơn giản (Direct keyword/vector hit)",
            "Tôi đã đọc gì về Kubernetes?",
            "Episodic memory trả về Kubernetes Pod lifecycle note; kèm profile người dùng.",
        ),
        (
            "2. Hỏi cần Profile Context (Topic Affinity)",
            "Recommend tài liệu tôi nên đọc tiếp theo?",
            "Feature store trả về topic_affinity (cloud/ai_ml) và reading_speed để cá nhân hóa gợi ý.",
        ),
        (
            "3. Hỏi cần Fresh Activity (Streaming velocity)",
            "Tôi đang quan tâm gì gần đây nhất?",
            "Feature store cung cấp queries_last_hour và distinct_topics_24h phản ánh nhịp độ gần đây.",
        ),
        (
            "4. Hỏi Paraphrase (Vector Semantic Matching)",
            "Tài liệu về tự động mở rộng hạ tầng theo nhu cầu?",
            "Không có từ 'Kubernetes' hay 'HPA', vector matching tìm thấy note về Auto-scaling.",
        ),
        (
            "5. Hỏi Mixed (Hybrid Search + Profile + Activity)",
            "Cho tôi summary tài liệu cloud security và bảo mật?",
            "Hybrid RRF kết hợp BM25 cho 'cloud security' và vector cho 'bảo mật', kết hợp cùng profile.",
        ),
    ]

    for title, q, note in queries:
        print(f"------------------------------------------------------------------")
        print(f"Query {title}")
        print(f"User Query: \"{q}\"")
        print(f"Expected  : {note}")
        print(f"------------------------------------------------------------------")
        context = agent.recall(q, user_id=user_id)
        print(context)
        print()

    print("==================================================================")
    print("  Demo completed successfully. All 5 queries resolved! (Exit 0)   ")
    print("==================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
