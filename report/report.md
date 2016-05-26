
## 2016年 Preferred Infrastructure / Preferred Networks　インターン選考コーディング課題

ヤマザキ裕幸ヴインセント hiroyuki.vincent.yamazaki@gmail.com

###  環境設定

Python 3.5

##### 依存ライブラリ（課題５）

`numpy`
`six`
`chainer`

## 実行手順

#### ファイル構造

```
$ASSIGNMENTS_ROOT \
	assignment1.py
	assignment2.py
	assignment3.py
	assignment4.py
	assignment4_adagrad.py
	assignment5.py
	utils \
		dataset.py
		randomizer.py
	output \
		assignment4_params
```

### 課題スクリプトの実行手順

```bash
cd $ASSIGNMENTS_ROOT
python assignment1.py // 課題１　外積を計算する関数のテストが実行されます
python assignment2.py // 課題２　アファイン変換を行う関数のテストが実行されます
python assignment3.py // 課題３　学習の１イテレーションが実行されます
python assignment4.py // 課題４　Autoencoderのモデルが学習されます
python assignment5.py // 課題５　Chainerで書かれたAutoencoderが学習されます
```

または `cd $ASSIGNMENTS_ROOT && ./test.sh` で全てのスクリプトが実行されます。

### 課題４

- **パラメータ**：`$ASSIGNMENTS_ROOT/output/assignment4_params`
- **平均ロス**：`0.11001077846872453`
- **追記**：ナイーブな実装でロスが`0.1`を下回らなかったのでAdaGradを実装したコードが`assignment4_adagrad.py`になります。しかし、最終的な平均ロスは`0.1100..`でほとんど変わりませんでした。 

## レポート（課題５）

## 過学習

今回の課題において、学習中のロスを計算するために用いられているValidation SetがTraining Setと全く一緒であるため、過学習（Overfitting）を起こしていると推測することができる。

<img src="aaa.png" style="width: 200px">

