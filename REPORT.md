# Lab 21 — Evaluation Report

**Học viên**: Khuất Văn Vương - 2A202600087
**Ngày nộp**: 2026-05-07  
**Submission option**: A + B

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, lấy mẫu 200 examples
- **Train/Eval split**: 180 train / 20 eval (seed = 42)
- **max_seq_length**: 1024 (thống kê token length: p50 = 227, p95 = 562, p99 = 704; làm tròn lên power-of-2 và cap ở 1024)
- **GPU**: Colab T4 16GB
- **LoRA config**:
  - target modules: `q_proj`, `v_proj`
  - dropout: 0
  - rank/alpha: r=8/16, r=16/32, r=64/128
- **Training config**:
  - 3 epochs, tổng 69 steps mỗi run
  - train batch size = 1, gradient accumulation = 8 (effective batch = 8)
  - learning rate = 2e-4, scheduler = cosine, warmup ratio = 0.10
  - optimizer: `adamw_8bit`
- **Training cost ước tính**:
  - Tổng thời gian train 3 rank: ~12.88 phút
  - Nếu dùng Colab Free: chi phí trực tiếp ~0 USD

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
| ---- | ---------------: | ---------------: | -------------: | --------: | ---------: |
| 8    |        1,843,200 |           4.2243 |        11.2155 |    1.5577 |     4.7479 |
| 16   |        3,686,400 |           4.4010 |        10.6146 |    1.5161 |     4.5544 |
| 64   |       14,745,600 |           4.2589 |        11.9980 |    1.4768 |     4.3790 |

**Nhận xét nhanh từ số liệu**:

- Rank tăng theo tuyến tính với số trainable params (r64 gấp 4 lần r16 và gấp 8 lần r8).
- PPL cải thiện dần khi tăng rank: r8 (4.7479) > r16 (4.5544) > r64 (4.3790).
- So với r16, r64 giảm thêm khoảng 3.85% perplexity.
- Mức VRAM của cả 3 run đều nằm trong ngưỡng T4 16GB; r64 cao nhất (~12.0GB).
- Train time giữa 3 rank chênh không đáng kể trong thí nghiệm này (~4.2–4.4 phút/run).

## 3. Loss Curve Analysis

Quan sát log huấn luyện cho cả 3 rank cho thấy training loss giảm ổn định theo step. Các checkpoint cuối không có dấu hiệu loss dao động mạnh hoặc diverge. Vì chiến lược eval trong lúc train được đặt là `eval_strategy = "no"` (để tránh OOM trên T4), tín hiệu overfitting theo nghĩa "train loss giảm nhưng eval loss tăng" không theo dõi được liên tục theo epoch; tuy nhiên eval cuối cùng sau train vẫn hợp lý và không có dấu hiệu suy giảm chất lượng bất thường.

Diễn giải theo rank:

- **r8**: học được nhưng độ fit còn hạn chế hơn hai rank còn lại.
- **r16**: điểm cân bằng tốt giữa chất lượng và tài nguyên.
- **r64**: loss/ppl tốt nhất, cho thấy năng lực biểu diễn cao hơn trên dataset hiện tại.

## 4. Qualitative Comparison (5 examples)

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Trả lời đúng ý chính nhưng diễn đạt lặp và chưa gọn.  
**Fine-tuned (r=16)**: Cấu trúc mạch lạc hơn, định nghĩa rõ hơn vai trò của dữ liệu và dự đoán.  
**Nhận xét**: Improved (clarity, readability).

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Có đưa code nhưng bị dừng giữa chừng và xử lý input chưa nhất quán.  
**Fine-tuned (r=16)**: Trả về phiên bản vòng lặp rõ ràng hơn, có kiểm tra input âm bằng exception.  
**Nhận xét**: Improved (completeness, safer input handling).

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Đúng hướng nhưng giải thích dài dòng.  
**Fine-tuned (r=16)**: Dạng liệt kê gọn hơn, có định hướng hành động.  
**Nhận xét**: Slightly improved (formatting concise), nhưng nội dung chuyên môn chưa thật sâu.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Có nêu đúng trọng tâm low-rank adaptation vs quantization-aware finetuning.  
**Fine-tuned (r=16)**: Có xu hướng dùng thuật ngữ chưa chuẩn ở một số chỗ ("Layer-wise Adaptive Regularization Optimization").  
**Nhận xét**: Mixed result; form tốt hơn nhưng factual precision cần kiểm chứng thêm.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Phân biệt được ba hướng tiếp cận ở mức khái quát.  
**Fine-tuned (r=16)**: Văn phong mượt hơn nhưng vẫn thiên mô tả tổng quan, chưa đi sâu case-by-case.  
**Nhận xét**: Slight improvement về diễn đạt; chiều sâu kỹ thuật tương đương.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, rank **r=64** cho chất lượng định lượng tốt nhất với perplexity thấp nhất (**4.3790**), cho thấy tăng rank giúp mô hình học biểu diễn tốt hơn trên tập dữ liệu 200 mẫu. Tuy vậy, mức cải thiện từ r16 lên r64 chỉ khoảng **3.85%** perplexity trong khi số trainable params tăng **4 lần** (3.69M lên 14.75M) và VRAM đỉnh cũng cao hơn. Điều này thể hiện quy luật diminishing returns: tăng rank vẫn có lợi, nhưng lợi ích biên bắt đầu nhỏ hơn nhiều so với chi phí tham số. Rank **r8** nhẹ nhất nhưng chất lượng thấp hơn rõ rệt (PPL 4.7479), phù hợp khi ưu tiên tiết kiệm tài nguyên hơn chất lượng. Với bối cảnh triển khai thực tế trên T4/chi phí hạn chế, em khuyến nghị dùng **r=16** làm lựa chọn ROI tốt nhất vì cân bằng chất lượng, VRAM và độ ổn định; chỉ chọn **r=64** khi bài toán đòi hỏi chất lượng tối đa và có ngân sách tài nguyên lớn hơn.

## 6. What I Learned

- Chất lượng dataset (instruction-output sạch, nhất quán) ảnh hưởng kết quả mạnh hơn việc chỉ tăng rank.
- Trên GPU nhỏ như T4, cấu hình huấn luyện an toàn (`eval_strategy=no`, `packing=False`, `adamw_8bit`, gradient accumulation) giúp tránh OOM và hoàn thành được đầy đủ thí nghiệm.
- Rank cao hơn không luôn mang lại ROI tốt hơn; cần đánh đổi giữa perplexity cải thiện và chi phí tham số/VRAM để chọn cấu hình production phù hợp.
