# MiniGPT (Character-level Transformer)

## 1. 项目简介
本项目实现了一个从零开始的字符级 Transformer（MiniGPT），包括：

- 数据预处理
- Tokenizer 构建
- Transformer Block（Multi-Head Attention + FFN）
- 训练与验证
- 自回归文本生成

---

## 2. 数据集
- 数据来源：自定义文本（约 3.8k 字符）
- 切分方式：90% train / 10% val
- Token 类型：字符级

---

## 3. 模型结构

- Embedding: token + position
- Transformer Block × 2：
  - Multi-head self-attention
  - FeedForward
  - Residual + LayerNorm
- 输出层：Linear → vocab

---

## 4. 训练设置

- block_size = 64
- n_embd = 128
- n_head = 4
- n_layer = 2
- optimizer = AdamW
- learning_rate = 3e-4

---

## 5. 实验结果

### Loss 曲线
（插入 loss_curve.png）

### 现象
- 前期 train/val loss 同步下降
- 后期出现过拟合（val loss 上升）

---

## 6. 推理示例

```
============================================================
Prompt: To be
------------------------------------------------------------
To bene, art red hird, ght chais pa sth hen wen,
And ts theivir bled hauwhereard er brendowhind.

Ther a crachedee sune rad beawarea p
Ithared fthat kin d.

============================================================
Prompt: The king
------------------------------------------------------------
The king.
Itr feer datrolst throm wan throors win-gh sasurveery ghthththt.

Asill the s thaind we mes wis atadibenemOrtld s wet wirinen wilever d,
And vithen 

============================================================
Prompt: If love
------------------------------------------------------------
If lovery.
Noe hillt os as nned warke the row thisle fingin arrowchee ch.
Th, fe me beeat ily chan nt menute stheawarermon ba,
Thep s berorol m wine wand t i

============================================================
Prompt: O my
------------------------------------------------------------
O my of kept, com.

A cand iAnde nis warand hiled grin bot.
Yepper disththik the hein id ak wbrnd, meat plen.
Son m bud bothent mo momom sivomnNomban,
The
```

---

## 7. 问题与局限

- 数据量太小
- 字符级建模难以捕捉语义
- 长距离一致性差
- 容易出现重复

---

## 8. 后续改进方向

- 更大数据集
- subword tokenizer
- 更深模型
- better sampling（top-k / temperature）

---

## 9. 关键理解

- attention score: [B, T, T]
- logits: [B, T, vocab]
- 训练：并行（teacher forcing）
- 推理：自回归（逐 token 生成）