### 前処理とデータ加工の履歴

- 機械学習ライブラリから、独自顔写真を分析対象にした。

- k-means、PCA、NMFをscikit-leansから分析できるようにサンプルを書いた。

- 通常の顔写真サンプルをグレースケールにして、分析できるようにした。

- 通常の機械学習を実行、値が3種類に変わるように見える。

- tensorflow/keras、学習モデルを作って出力した。(検証中)

  ※ 環境に応じて、モデルの内容を調整する必要あり。

- 多対多の顔認識、主成分分析を用いて近似値評価のための中央値を計算で導いた。

  ※  用途として、最小値と最大値をIF文などの条件式で設定し使用する。

- プロジェクト内で実行できるようにライブラリ化をした。

```markdown
# ファイルを実行
cd rock_ptamigan/lib

python main.py
```

```markdown
# 出力結果
Approximate value : 0.15 in train folder.
Approximate value : 0.11 in validation folder.
```
