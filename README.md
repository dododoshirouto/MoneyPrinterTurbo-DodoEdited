# MoneyPrinterTurbo（カスタムフォーク）

> **このリポジトリは [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) v1.3.0 をベースにしたフォークです。**
> オリジナルの設計思想・構造を維持しつつ、日本語運用・ローカルLLM・VOICEVOX・背景動画合成・GPU対応などを中心に独自の機能拡張を行っています。

---

## 概要

テキストから縦型ショート動画を自動生成するツールです。スクリプト生成・音声合成・動画素材収集・字幕・エフェクト合成までをワンクリックで完結させます。

---

## オリジナルからの主な変更点

### LLM・スクリプト生成

- **llama.cpp（llmcpp）対応** — ローカルLLMサーバーをプロバイダとして利用可能
- **LLMプリセット保存/読込** — プロバイダ・モデル名・プロンプト等をプリセットとして保存・切替
- **タイトル自動生成** — スクリプトから15文字前後のキャッチーなタイトルをLLMで自動生成
- **タイトルをスクリプト推論に含めるオプション** — 既存タイトルがある場合はそれを参照してスクリプトを生成
- **色強調タグ付き生成** — スクリプト中の重要語句にLLMが `<color1>`/`<color2>`/`<color3>` タグを自動付与
- **Web検索推論** — スクリプト生成前にWebを自動検索して情報収集。DuckDuckGo（APIキー不要）/ Brave / SerpAPI を選択可能。オーケストレーターLLMが検索戦略を立案し、ワーカーLLMが結果を要約する2スレッド構成。ステップ数はGUIで設定し、調査ログは `research.json` に保存

### 音声合成

- **VOICEVOX対応** — `voicevox:<speaker_id>:<名前>` 形式で話者を選択
- **AivisSpeech対応** — 同様の形式でローカル音声合成エンジンを利用可能

### 動画合成

- **背景動画レイヤー** — メイン動画（contain）の背後に背景動画（cover）を配置する2レイヤー合成
  - 背景動画はピクサベイ等の素材ソースから自動取得またはカスタム指定
  - `combine_videos` 内部で時刻オフセットを持つ背景スライスを合成するため、黒帯が完全に消える
- **動画フィットモード** — contain / cover を選択可能
- **カバーサイズ上下マージン** — cover表示時のセーフマージンを割合で指定
- **テキストマージン** — 字幕・タイトルの左右/上下マージンを割合で指定

### 字幕・テキスト

- **色強調字幕** — `<color1>〜</color3>` タグで文字色と縁取り色を個別に設定（各色のオン/オフ・色選択可）
- **タイトルオーバーレイ** — 動画上部にタイトルテキストを常時表示（字幕とは独立して配置）
- **字幕タイムシフト** — 字幕表示タイミングをオフセット調整可能
- **OTF/TTCフォント対応** — OTF/TTCフォントも使えるようにする

### UI・設定

- **日本語UI対応** — 表示言語に日本語を追加
- **タイムゾーン設定** — `config.toml` でタイムゾーンを指定（ログ・フォルダ名の日時に反映）
- **タスクフォルダ名テンプレート** — Jinja2形式で日時・モデル名・シード値などを組み合わせて命名
- **scripts.json拡張** — タスク生成時のすべてのパラメータ（テーマ・音声設定・プロンプト等）を記録
- **ワンクリック生成** — GUIトップのテーマ入力＋ボタン1つで「Web検索→スクリプト生成→キーワード生成→動画生成」を全自動実行
- **プリセット エクスポート/インポート** — 選択中のプリセットをJSONファイルで書き出し・読み込み（1プリセット単位での共有・バックアップに対応）

### インフラ・パフォーマンス

- **NVIDIAハードウェアエンコード（NVENC）対応**
  - Dockerfile: apt版ffmpegを廃止し、BtbN静的GPLビルド（h264_nvenc/hevc_nvenc含む）をインストール
  - docker-compose: `runtime: nvidia` + `deploy.resources.reservations.devices` でGPUをコンテナに渡す
  - `IMAGEIO_FFMPEG_EXE` 環境変数でMoviePyが使うffmpegバイナリを明示指定
  - `config.toml` の `video_codec = "h264_nvenc"` でNVENCエンコードが有効になる

---

## セットアップ

### 前提条件

- Docker / Docker Compose
- NVIDIA GPU使用時: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# nvidia-container-toolkit のインストール（Ubuntu/WSL2）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 起動

```bash
git clone <このリポジトリ>
cd MoneyPrinterTurbo
docker compose up --build -d
```

ブラウザで `http://127.0.0.1:8501` を開く。

### NVENCを有効にする

`config.toml` の `[app]` セクションに追加：

```toml
[app]
video_codec = "h264_nvenc"
```

---

## 設定ファイル

| ファイル | 用途 |
|---|---|
| `config.toml` | アプリ全体の設定（LLMプロバイダ・音声・エンコーダ等） |
| `storage/presets.json` | GUIプリセット保存先 |
| `resource/fonts/` | 字幕・タイトル用フォント |
| `resource/bg_video/` | カスタム背景動画の配置先 |
| `resource/songs/` | BGM配置先 |

---

## ライセンス

オリジナル: [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) — MIT License

このフォークもMITライセンスに準拠します。
