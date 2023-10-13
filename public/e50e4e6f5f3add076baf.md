---
title: YouTube動画から文字起こしする簡易App
tags:
  - Flask
  - Docker
  - youtube-dl
  - SpeechRecognition
private: false
updated_at: '2021-08-19T00:36:25+09:00'
id: e50e4e6f5f3add076baf
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに
　YouTubeのインタビュー動画からテキスト抽出を行う必要が出てきたため、どうせなら簡単にできる様、ツール化しておきたいなと思って作成しました。あくまでも個人用なので完成度は低いですが、同じようなことやりたい人案外いるかな？と思ったので記事にまとめておきました。

- 個人利用の範囲を超える場合において、YouTubeの規約に違反する可能性があります。使用ライブラリ、及びYouTubeのポリシーをご確認ください。
- 個人利用の範囲内であったとしても、良識の範囲内での使用が求められます。サーバー負荷なども考慮してご使用ください。
- ソースコードは [[GitHubレポジトリ]](https://github.com/spider-man-tm/youtube-speech-to-text) を参照ください。

# 使用方法
- docker環境を想定しています。
- 上記リポジトリをローカルにクローン

```
$ git clone git@github.com:spider-man-tm/youtube-speech-to-text.git
```

- Dockerコンテナをビルド&起動

```
$ docker-compose up -d
```

- `http://localhost:5000`にアクセス
- 検索対象のURLを入力し、変換ボタンを押下

![image1.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/323251/005e2573-8fcc-18a8-1015-aeb1facd7900.gif)

- 処理中はボタンの押下ができなくなります。（ボタン連打防止）
- 処理結果が画面表示されます。また`app/log`に簡易ログ、`app/movies`にダウンロードされた動画ファイル、`app/text`に文字起こしされたテキストファイルがそれぞれ出力されます。

![image2.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/323251/941559a5-4a69-065f-def7-2d6d44779960.png)

- 上記結果はTOKYO2020で金メダルを獲得したソフトボールの上野由岐子選手のインタビュー動画を文字起こしした結果です。
- 精度は動画のクオリティによりまちまちでした。女性アナウンサーが読み上げるニュースなどはかなり正確ですが、BGMが大きかったりすると結構きつい印象です。
- 正しくないURLが入力された場合、変換できません。

![image3.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/323251/675de6a3-977e-4311-27e3-2c5a4056a100.gif)

# 補足
- 600秒経過しても処理が終了しない場合、強制的に処理を中断するようにしています。`multiprocessing `を使い、マルチプロセス化することで実装しています。

![スクリーンショット 2021-08-18 22.25.36.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/323251/6cd943a8-49dd-1955-e89e-beb13f4bd895.png)

```python
def main(url: str) -> None:
    try:
        with Pool() as p:
            apply_result = p.apply_async(process, (url,))
            is_finish = apply_result.get(timeout=600)
        return is_finish

    except TimeoutError:
        logger.error('process() Timeout!')
        return ('タイムアウト', '')
```
