
## 2016年 Preferred Infrastructure / Preferred Networks<br>インターン選考コーディング課題
ヤマザキ裕幸ヴインセント<br>hiroyuki.vincent.yamazaki@gmail.com

##  環境設定
- 環境 `Python 3.5`
- 依存ライブラリ（課題５）`numpy`、`six`、`chainer`


## 実行手順
```bash
cd $ASSIGNMENTS_ROOT
python assignment1.py 		// 課題１　外積関数のテスト
python assignment2.py 		// 課題２　アファイン変換のテスト
python assignment3.py 		// 課題３　学習の１イテレーション
python assignment4.py 		// 課題４　Autoencoderの学習
python assignment5.py 		// 課題５　Autoencoder（Chainer使用）の学習
```

`$ASSIGNMENTS_ROOT`はプロジェクトのルートディレクトリです。<br>
`cd $ASSIGNMENTS_ROOT && ./test.sh`でユティリティコードのテストを含めた、以上の全スクリプトを実行します。

**課題１−３**<br>
`assignment1.py`、`assignment2.py`、`assignment3.py`の実行、またはコードをご参照ください。
  
**課題４**<br>`assignment4.py`を10000イテレーション学習させた結果が以下の通りです。平均ロス `0.11002517934923145`。パラメータ `$ASSIGNMENTS_ROOT/output/assignment4_params`。追記：学習パラメータを変えても平均ロスが`0.1`以下まで減らなかったためAdaGradを実装したのですが、結果、平均ロスはほぼ変わりませんでした。 そちらのコードが`$ASSIGNMENTS_ROOT/assignment4_adagrad.py`になります。
	
**課題５**<br>
`assignment5.py`の実行、またはコード、またはレポートをご参照ください。