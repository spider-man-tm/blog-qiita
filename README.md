Blog Qiita
===

### Qiita URL
https://qiita.com/Takayoshi_Makabe <br />
※ マイページへのリンクです

### 📘 How to use
Qiita CLI: https://qiita.com/Qiita/items/666e190490d0af90a92b <br />
Qiita Repo: https://qiita.com/Qiita/items/32c79014509987541130

#### 基本操作

- プレビュー画面の表示

```shell
# qiita.config.jsonで指定したport
npx qiita preview
```

- 新規空記事の作成

```shell
npx qiita new 記事のファイルのベース名
```

- 記事の更新・投稿
  - `main`・`master`ブランチにpushすれば反映される
  - それ以外に、コマンドで更新する場合は以下の通り

```shell
# 1部の記事のみ
npx qiita publish 記事のファイルのベース名

# 全ての記事
npx qiita publish --all
```

- 記事の削除
  - Qiita CLI、Qiita Preview から記事の削除はできない
  - publicディレクトリから markdown ファイルを削除しても Qiita 上では削除はされない
  - Qiita上で記事の削除を行う
