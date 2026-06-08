## Tutorial: Generative AI Meets Reinforcement Learning - ICML 2025

> _Benjamin Eysenbach (Princeton) - Amy Zhang (UT Austin)_ _https://generative-rl-tutorial.github.io/_

----------

## Mục Lục

**Phần I - Tổng Quan Chủ Đề**

1.  [Bài Toán Đặt Ra](#bài-toán-đặt-ra)
2.  [Động Lực Nghiên Cứu](#động-lực-nghiên-cứu)
3.  [Tầm Quan Trọng của Hướng Nghiên Cứu](#tầm-quan-trọng-của-hướng-nghiên-cứu)

**Phần II - Kiến Thức Nền** 

  1. [Reinforcement Learning là gì?](#reinforcement-learning-là-gì) 
  2. [Generative Models là gì?](#generative-models-là-gì)
  3. [Xác Suất & Phân Phối - Nền Tảng Toán Học](#xác-suất-phân-phối-nền-tảng-toán-học)
  4. [Sự giao thoa giữa RL và Generative Modeling](#sự-giao-thoa-giữa-rl-và-generative-modeling)

----------

# PHẦN I - TỔNG QUAN CHỦ ĐỀ

----------

## 1. Bài Toán Đặt Ra

### 1.1 Giới Hạn Cốt Lõi của AI Hiện Tại: Vấn Đề Bắt Chước

Năm 1770, Wolfgang von Kempelen giới thiệu với châu Âu một cỗ máy biết chơi cờ - "Mechanical Turk" - một cỗ máy với thiết kế nhiều bánh răng tinh vi. Nó đánh bại Napoleon, Benjamin Franklin, và vô số người chơi khác. Hóa ra, các bánh răng trong máy chỉ là bình phong, và có một người thật ẩn bên trong, điều khiển mọi nước đi qua gương phản chiếu.

Câu chuyện này là ẩn dụ chính xác cho trạng thái AI ngày nay:

> _"Các mô hình generative hiện tại là kỳ tích kỹ thuật ấn tượng, và khiến nhiều người nghĩ chúng thực sự thông minh. Nhưng khi bóc lớp vỏ ra, bạn nhận ra chúng thường không thông minh hơn những con người được lấy làm mẫu để train."_

**GPT-4, Gemini, Stable Diffusion** - tất cả đều được huấn luyện bằng một nguyên tắc chung: **maximum likelihood imitation (MLE)** - tối đa hóa xác suất tạo ra output giống với dữ liệu của con người. Đây là phương pháp cực kỳ mạnh, nhưng chứa đựng một giới hạn nền tảng:

```
Khả năng của mô hình ≤ Khả năng của dữ liệu huấn luyện
```

Nếu tất cả văn bản con người viết đều sai, mô hình sẽ sai theo. Nếu chưa con người nào giải được một bài toán nào đó, mô hình không thể học cách giải nó từ dữ liệu. 

Tutorial đặt ra hai câu hỏi liên quan chặt chẽ:

- Làm thế nào để xây dựng AI có thể vượt qua khả năng của dữ liệu huấn luyện - giải các bài toán mà chưa có con người nào biết cách giải?

- Generative AI và Reinforcement Learning - hai lĩnh vực tưởng như tách biệt, thực ra có cùng cấu trúc toán học không? Và nếu có, chúng ta có thể khai thác điều đó như thế nào?


### 1.2 Cấu trúc tutorial

Bốn phần của tutorial được trình bày theo logic tuyến tính:

**Phần I** - Generative models dùng trong RL (world model, policy)
**Phần II** - Tương tác giữa agent với môi trường trong RL là một generative process
**Phần III** - Tính xác suất của generative process trên, và các bài toán RL tổng quát
**Phần IV** - Nguồn gốc của dữ liệu, và xây dựng generative model tự thu thập dữ liệu

----------

## 2. Động Lực Nghiên Cứu

### 2.1 Giới hạn của chất lượng dữ liệu

Như đã nói ở phần trước về bài toán đặt ra, phần lớn AI hiện đại - từ ChatGPT đến Midjourney - được train bằng cách **tối đa hóa khả năng sinh lại dữ liệu của con người**. Điều này hiệu quả khi:

-   Có đủ dữ liệu chất lượng cao
-   Nhiệm vụ nằm trong phân phối dữ liệu

Nhưng gặp phải vấn đề khi:

-   Cần sáng tạo vượt ngoài dữ liệu
-   Bài toán chưa ai giải được (khoa học mới, chiến lược game mới, kỹ thuật robot mới)
- **AI models collapse when trained on recursively generated data** - khi AI train trên dữ liệu do AI sinh ra, chất lượng suy giảm theo cấp số nhân (Shumailov et al., _Nature_ 2024)

### 2.2 Giới hạn của Reward Engineering
 
Nền tảng của RL là sự phụ thuộc vào reward. Nhưng hầu hết bài toán thực tế **rất khó thiết kế reward function**, đòi hỏi kiến thức chuyên sâu và thậm chí phải biết trước policy tối ưu trông như thế nào.
 
Đây là mâu thuẫn cốt lõi:  RL hứa hẹn tìm ra giải pháp tốt hơn con người, nhưng reward function lại đòi hỏi con người phải biết nhiều về giải pháp đó để thiết kế.
 
### 2.3 Giới hạn phụ thuộc vào con người
 
Cả Supervised Learning lẫn RL đều phụ thuộc vào dữ liệu. Nhưng:
- **Supervised Learning:** dữ liệu được curate bởi con người -> bị giới hạn bởi khả năng con người
- **RL truyền thống:** cần reward function -> vẫn phụ thuộc vào con người

Liệu AI có thể tự xây dựng kiến thức của mình - tự thu thập dữ liệu, tự tạo curriculum, tự cải thiện - mà không cần giám sát liên tục từ con người?

### 2.4 Hai Lĩnh Vực Đang Hội Tụ

Trong khi RL và Generative AI phát triển song song trong nhiều thập kỷ, gần đây xuất hiện nhận thức rằng chúng đang giải quyết cùng một bài toán nhìn từ các góc khác nhau.

Nhận ra sự tương đương này mở ra một loạt kỹ thuật có thể được giao thoa giữa hai lĩnh vực.

Tutorial này xem xét ba hướng giao thoa chính:

| Hướng | Ý nghĩa |
|---|---|
| **GenAI -> RL** | Dùng generative models làm thành phần trong RL |
| **RL -> GenAI** | Nhìn RL như một generative process |
| **RL để train GenAI** | Dùng RL để cải thiện cách huấn luyện generative models |

----------

## 3. Tầm Quan Trọng của Hướng Nghiên Cứu

### 3.1 Ứng Dụng Thực Tiễn Ngay Trước Mắt

**Robotics:** Generative models như diffusion policies đã đạt SOTA cho robot manipulation. Kết hợp với RL, robot có thể vượt qua khả năng của demonstrations.

**Foundation Models cho hành vi:** Giống như GPT là foundation model cho ngôn ngữ, **Behavioral Foundation Models** (học từ RL không có reward) có thể là foundation model cho hành động - một model dùng cho nhiều task robot khác nhau mà không cần train lại.

**RLHF và alignment:** Reinforcement Learning from Human Feedback (RLHF) - kỹ thuật cốt lõi làm cho ChatGPT "nghe lời" - là ứng dụng trực tiếp của RL để fine-tune generative models. Hiểu sâu mối liên hệ này giúp thiết kế các phương pháp alignment tốt hơn.

**Khoa học và khám phá:** AlphaFold dự đoán cấu trúc protein; AlphaProof giải bài thi toán Olympic. Các hệ thống này dùng RL (tự chơi, feedback từ toán học) để tìm giải pháp mà không có dữ liệu tương ứng.




### 3.1 Generalization - Frontier Còn Bỏ Ngỏ
 
Notes so sánh với mô hình láng giềng gần nhất (nearest neighbor): 
>Nó không được coi là "thông minh" vì chỉ nhớ dữ liệu đã thấy. Điều khiến deep learning ấn tượng chính là khả năng dự đoán đúng trên những input chưa thấy bao giờ - generalization qua những gì học được.
 
Generative image models và language models đã đạt được điều này trong lĩnh vực của chúng. Nhưng RL agents ngày nay vẫn chủ yếu **ghi nhớ** thay vì generalize. Tutorial đặt câu hỏi:
 
> *"Tương tự như generative image models có thể tạo ra ảnh sống động sau khi chỉ train trên một phần nhỏ các ảnh có thể, liệu ta có thể xây dựng RL agents có thể xây lâu đài cát như thật sau khi chỉ practice một phần nhỏ các loại lâu đài cát?"*
 
Ví dụ từ notes: bạn có thể điều hướng hiệu quả khắp Manhattan dù chỉ đi qua một phần rất nhỏ trong 10¹⁷ tuyến đường có thể — vì bạn đã học các **khái niệm** (đường một chiều, ô vuông, tên đường) có thể transfer sang tuyến mới. RL cần năng lực tương tự.
 
### 3.2 Compression như Mục Tiêu Thống Nhất
 
Notes đề xuất một cách nhìn thống nhất cho cả RL và generative modeling qua lăng kính **nén thông tin (compression)**:
 
| Loại học | Đối tượng nén | Kết quả |
|---|---|---|
| Supervised / Unsupervised Learning | Dataset X | Model có thể generalize trên X |
| Self-supervised RL | MDP (môi trường không có reward) | Behavioral Foundation Model |
 
Nén một MDP - tìm ra các pattern cấu trúc trong không gian hành vi - chính là điều skill learning và successor representations đang làm. Học các pattern đó giúp agent generalize sang task mới giống như biểu diễn tốt giúp classifier generalize sang ảnh mới.
 
### 3.3 Tiềm năng của GenAI + RL
 
Tutorial kết thúc với một nhận định thực tế về khoảng cách giữa hiện tại và tiềm năng:
 
> *"Mặc dù có tiến bộ nhanh chóng trong RL một thập kỷ qua, các thuật toán ngày nay vẫn chỉ giữ lại một phần rất nhỏ năng lực tiềm năng của RL."*
 
Trong khi generative models ngày nay "chỉ vẽ pixel và token lên màn hình", các hệ thống tương lai - nếu kết hợp được RL và generative modeling đúng cách - có thể **xây dựng thế giới** mà generative models hiện tại chỉ mô tả được.

----------

# PHẦN II - KIẾN THỨC NỀN

----------

## 1. Reinforcement Learning là gì?

### 1.1 Định Nghĩa Bài Toán RL

Bài toán RL được mô tả bởi một **Markov Decision Process (MDP)**, bao gồm 5 thành phần:

```
MDP = (S, A, p, r, γ)
```

| Ký hiệu | Tên | Ý nghĩa | Ví dụ (game Tetris) |
|---|---|---|---|
| **S** | State space | Tập tất cả trạng thái có thể | Tất cả cấu hình bảng Tetris có thể |
| **A** | Action space | Tập tất cả hành động có thể | {trái, phải, xoay, thả} |
| **p(s'\|s,a)** | Dynamics | Xác suất chuyển sang s' khi làm a ở s | Xác suất khối rơi vào ô nào | 
| **r(s,a)** | Reward function | Phần thưởng tức thời | +1 mỗi dòng xóa được |
| **γ ∈ [0,1)** | Discount factor | Trọng số phần thưởng tương lai | Thường từ 0.9-0.999 |

 **Tính Markov**  là giả định quan trọng: state hiện tại chứa _đủ thông tin_ để quyết định hành động tiếp theo - không cần nhớ lịch sử. Ví dụ: vị trí hiện tại trên bảng Tetris chứa đủ thông tin, không cần biết lịch sử các khối đã rơi.

### 1.2 Policy

**Policy** π là hàm ánh xạ từ state sang action (hoặc phân phối trên action):

```
π(a | s) = xác suất chọn action a khi ở state s
```

Ví dụ với Gaussian policy :

```
π_θ(a | s) = N(a; μ_θ(s), σ²I)
           = policy với mean μ_θ(s) được tính bởi neural network,
             và variance σ² cố định
```
**Trong đó:**
- $\pi_\theta(a | s)$: Xác suất chọn hành động $a$ khi đang ở trạng thái $s$.
- $\mathcal{N}(\dots)$: Ký hiệu của Phân phối chuẩn (Hình chuông). Nó nói rằng các hành động được chọn sẽ phân bố giống như hình quả chuông úp ngược - tập trung nhiều ở giữa và ít dần về hai bên.
- $\mu_\theta(s)$: Tâm của hình chuông (Hành động lý tưởng nhất do mạng neural tính toán ra).
- $\sigma^2 I$: Độ rộng của hình chuông (Độ ngẫu nhiên).

Hành động mà policy lựa chọn là một phân phối chứ không cố định vì:
-   Exploration: output phân phối cho phép agent thử các action khác nhau, đôi khi giúp tìm ra chiến thuật tối ưu hơn
-   Uncertainty: trong môi trường ngẫu nhiên, hành động ngẫu nhiên khiến AI thực tế hơn và "giống con người hơn"
-   Continuity: phân phối dễ tối ưu hơn bằng gradient descent

### 1.3 Trajectory và Return

**Trajectory** τ = (s₀, a₀, s₁, a₁, ...) là chuỗi state-action theo thời gian.

**Discounted Return** R(τ) là tổng phần thưởng có chiết khấu:

```
R(τ) = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + ...
     = Σₜ γᵗ · r(sₜ, aₜ)
```

**Tại sao cần discount factor γ?**

Discount factor giải quyết hai vấn đề:

_Vấn đề toán học:_ Không có γ < 1, tổng phần thưởng có thể vô hạn (khi horizon vô hạn).

_Vấn đề thực tế:_ Phần thưởng gần hơn nên có giá trị hơn - đây là nguyên tắc phổ quát trong kinh tế học (time value of money). γ = 0.99 nghĩa là phần thưởng sau 100 bước chỉ đáng bằng e^(-1) ≈ 37% phần thưởng ngay bây giờ.

```
γ = 0.0 -> Agent hoàn toàn myopic (chỉ quan tâm reward hiện tại)
γ = 0.9 -> Tầm nhìn ~10 bước
γ = 0.99 -> Tầm nhìn ~100 bước
γ -> 1.0 -> Tầm nhìn vô hạn (average reward setting)
```

### 1.4 Mục Tiêu RL

**Mục tiêu:** Tìm policy π tối đa hóa expected discounted return:

```
J(π) = E_{τ ~ π} [Σₜ γᵗ · r(sₜ, aₜ)]
     = E_{τ ~ π} [R(τ)]
```

Ký hiệu `E_{τ ~ π}[·]` nghĩa là kỳ vọng lấy trên tất cả trajectory mà policy π có thể tạo ra.

**So sánh với mục tiêu Supervised Learning:**

```
Supervised Learning:   max_θ E_{(s,a) ~ D}[log π_θ(a | s)]
                      (tối đa likelihood trên dữ liệu cố định D)

Reinforcement Learning: max_θ E_{τ ~ π_θ}[R(τ)]
                       (tối đa return trên trajectories do π sinh ra)
```

**Điểm khác biệt căn bản:** Trong RL, dữ liệu (trajectories) phụ thuộc vào chính policy ta đang học - không có dataset cố định. Đây vừa là điểm mạnh (có thể khám phá), vừa là thách thức (non-stationary data).

### 1.5 Value Functions - Ước Lượng Return Tương Lai

Để tối ưu J(π), ta cần biết "từ state này, action nào dẫn đến return cao nhất?" - đây là vai trò của **value functions**.

**State Value Function** V^π(s): Return kỳ vọng khi xuất phát từ state s và theo policy π:

```
V^π(s) = E_π[Σₜ γᵗ · r(sₜ, aₜ) | s₀ = s]
```

**State-Action Value Function** Q^π(s, a): Return kỳ vọng khi xuất phát từ (s, a) và sau đó theo policy π:

```
Q^π(s, a) = E_π[Σₜ γᵗ · r(sₜ, aₜ) | s₀ = s, a₀ = a]
```

**Ý nghĩa trực quan của Q-function:**

-   Q(s, a) cao -> thực hiện action a ở state s dẫn đến tương lai tốt
-   Q(s, a) thấp -> action a ở state s dẫn đến tương lai tồi
-   Chính sách tối ưu: chọn `a* = argmax_a Q*(s, a)`

**Tại sao Q-function quan trọng cho tutorial này:** Q-function sẽ xuất hiện xuyên suốt - trong diffusion policies (dùng Q để guide hoặc select), trong temporal contrastive learning (Q ≈ log likelihood ratio), trong zero-shot RL (Q tính từ biểu diễn học được).

### 1.6 Bellman Equations - Quan Hệ Đệ Quy

**Bellman Equation** là quan hệ đệ quy cơ bản của RL:

```
Q^π(s, a) = r(s, a) + γ · E_{s' ~ p(·|s,a), a' ~ π(·|s')} [Q^π(s', a')]
           = phần thưởng ngay lập tức + γ · Q kỳ vọng ở bước tiếp
```

**Ý nghĩa:** Return từ bước hiện tại = reward bây giờ + phần còn lại (bị chiết khấu).

**Bellman Optimality Equation** cho policy tối ưu:

```
Q*(s, a) = r(s, a) + γ · E_{s'} [max_{a'} Q*(s', a')]
```

**Tại sao quan trọng cho tutorial:** Bellman equation cho phép học Q-function từ dữ liệu off-policy (Temporal Difference learning). Trong tutorial, biến thể của nó — **Bellman flow constraint** - là nền tảng cho toàn bộ Phần III.

### 1.7 Policy Gradient - Cách tối ưu Policy

Để tối đa hóa J(π), ta lấy gradient theo tham số θ của policy:

```
∇_θ J(π_θ) = E_{τ ~ π_θ} [R(τ) · Σₜ ∇_θ log π_θ(aₜ | sₜ)]
```

**Trong đó:**

-   `R(τ)` = trọng số: trajectory tốt (reward cao) -> cập nhật nhiều hơn
-   `∇_θ log π_θ(aₜ | sₜ)` = score function: hướng để policy "thích" action aₜ hơn

**Trực quan:** Policy gradient nói: _"Hãy làm những gì đã dẫn đến return cao nhiều hơn, và làm những gì dẫn đến return thấp ít hơn."_

**Score function** `∇_θ log π_θ(a | s)` xuất hiện lại trong generative modeling — đây là cầu nối giữa RL và diffusion models (score-based generative models). Đây là một trong những kết nối trung tâm của tutorial.

----------

## 2. Generative Models là gì?

----------
## 3. Xác Suất & Phân Phối - Nền Tảng Toán Học
----------

## 4. Sự giao thoa giữa RL và Generative Modeling

Dưới đây là các khái niệm tương đương giữa RL và Generative Modeling - hiểu bảng này giúp đọc tutorial dễ hơn:

| Reinforcement Learning | Generative Modeling | Ý nghĩa chung |
|---|---|---|
| Policy π(a\|s) | Generative model p(x) | Phân phối cần học |
| Score function ∇_θ log π(a\|s) | Score function ∇_x log p(x) | Gradient của log-prob |
| Return R(τ) | Log-likelihood log p(x) | Tín hiệu học |
| Trajectory τ | Latent variable z | Biến tiềm ẩn |
| Policy gradient | Score function estimator | Cách tối ưu phân phối |
| Q-function Q(s,a) | Likelihood ratio p^π(sf\|s,a)/p(sf) | Ước lượng xác suất tương lai |
| Occupancy measure d^π(s,a) | Data distribution p(x) | Phân phối ta muốn matching |
| Bellman equation | Markov property | Cấu trúc đệ quy |
| DICE regularizer | f-divergence giữa phân phối | Giữ gần phân phối tham chiếu |
| Skill z | Latent code z trong VAE | Điều khiển hành vi/output |

----------
