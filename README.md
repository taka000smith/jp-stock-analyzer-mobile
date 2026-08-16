# 日本株 ファンダ＋テクニカル分析アプリ

## セットアップ
```bash
pip install -r requirements.txt
streamlit run app.py
```

入力例: `7203, 6758, 8035`

Version 1はyfinanceを使用します。日本株の財務データは銘柄によって欠損があります。判定・Entry・損切りは学習/スクリーニング用の参考値で、最終判断前に決算短信・決算説明資料・TDnet等の一次情報を確認してください。
