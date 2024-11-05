

import whisper

import numpy as np
import contextlib
import wave
#from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment, notebook

from scipy.spatial.distance import cosine
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple

import csv
import pandas as pd


#-------------- return embeddings --------------
def segment_embedding(
    file_name: str,
    duration: float,
    segment,
    embedding_model: PretrainedSpeakerEmbedding
) -> np.ndarray:
    """
    音声ファイルから指定されたセグメントの埋め込みを計算します。
    
    Parameters
    ----------
    file_name: str
        音声ファイルのパス
    duration: float
        音声ファイルの継続時間
    segment: whisperのtranscribeのsegment
    embedding_model: PretrainedSpeakerEmbedding
        埋め込みモデル

    Returns
    -------
    np.ndarray
        計算された埋め込みベクトル
    """
    audio = Audio()
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(file_name, clip)
    #print(waveform)
    return embedding_model(waveform[None])

def generate_speaker_embeddings(
    meeting_file_path: str,
    transcript
) -> np.ndarray:
    """
    音声ファイルから話者の埋め込みを計算します。
    
    Parameters
    ----------
    meeting_file_path: str
        音声ファイルのパス
    transcript: Whisper API の transcribe メソッドの出力結果

    Returns
    
    -------
    np.ndarray
        計算された話者の埋め込み群
    """
    segments = transcript['segments']
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device='cuda')
    embeddings = np.zeros(shape=(len(segments), 192))

    with contextlib.closing(wave.open(meeting_file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(meeting_file_path, duration, segment, embedding_model)

    embeddings = np.nan_to_num(embeddings)
    return embeddings


#-------------- match by cosine similarity --------------
def closest_reference_speaker(embedding: np.ndarray, references: List[Tuple[str, str, np.ndarray]]):
    """
    与えられた埋め込みに最も近い参照話者を返します。

    Parameters
    ----------
    embedding: np.ndarray
        話者の埋め込み
    references: List[Tuple[str, str, np.ndarray]]
        参照話者の名前，性別と埋め込みのリスト

    Returns
    -------
    list
        最も近い参照話者の名前と性別
    """
    min_distance = float('inf')
    closest_speaker = []
    for name, sex, reference_embedding in references:
        #print(reference_embedding.shape)
        reference_embedding = reference_embedding[0]
        distance = cosine(embedding, reference_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_speaker = [name, sex]

    return closest_speaker

# def format_speaker_output_by_segment2(embeddings: np.ndarray, transcript: dict, reference_embeddings: List[Tuple[str, np.ndarray]]) -> str:
#     """
#     各発話者の埋め込みに基づいて、セグメントを整形して出力します。

#     Parameters
#     ----------
#     embeddings: np.ndarray
#         話者の埋め込みのリスト
#     transcript: dict
#         Whisper API の transcribe メソッドの出力結果
#     reference_embeddings: List[Tuple[str, np.ndarray]]
#         参照話者の名前と埋め込みのリスト

#     Returns
#     -------
#     str
#         発話者ごとに整形されたセグメントの文字列。
#     """
#     labeled_segments = []
#     for embedding, segment in zip(embeddings, transcript["segments"]):
#         #print(embedding.shape)
#         speaker_name = closest_reference_speaker(embedding, reference_embeddings)
#         labeled_segments.append((speaker_name, segment["start"], segment["text"]))

#     output = ""
#     for speaker, _, text in labeled_segments:
#         output += f"{speaker}: 「{text}」\n"
#     return output

def format_speaker_output_by_segment2(embeddings: np.ndarray, transcript: dict, reference_embeddings: List[Tuple[str, str, np.ndarray]]) -> pd.DataFrame:
    """
    各発話者の埋め込みに基づいて、セグメントを整形して出力します。

    Parameters
    ----------
    embeddings: np.ndarray
        話者の埋め込みのリスト
    transcript: dict
        Whisper API の transcribe メソッドの出力結果
    reference_embeddings: List[Tuple[str, str, np.ndarray]]
        参照話者の名前，性別と埋め込みのリスト

    Returns
    -------
    pd.DataFrame
        各セグメントの文字列，名前，性別を含むデータフレーム．
    """
    # Fill the DataFrame with speaker info
    output_df = pd.DataFrame(columns=["text", "name", "sex"])
    for embedding, segment in zip(embeddings, transcript["segments"]):
        #print(embedding.shape)
        speaker_info = closest_reference_speaker(embedding, reference_embeddings)
        output_df = pd.concat([output_df, pd.DataFrame([[segment["text"], speaker_info[0], speaker_info[1]]], columns=["text", "name", "sex"])])


    # Merge consecutive segments by the same speaker
    merged_rows = []
    
    for i in range(len(output_df)):
        if i > 0 and output_df.iloc[i]["name"] == output_df.iloc[i - 1]["name"]:
            # If the current name matches the previous one, concatenate the text
            merged_rows[-1]["text"] += " " + output_df.iloc[i]["text"]
        else:
            # Otherwise, add the current row to merged_rows
            merged_rows.append(output_df.iloc[i].copy())

    # Create a new DataFrame from merged rows
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)

    return merged_df
    


#-------------- (main) transcribe and voice matching --------------
# 実行時間　約48秒

model = whisper.load_model("large")

# 文字起こしする音声
# download audioURL from web app. we necessary to convert to wav.
FILE_PATH = "./resource/record241023/meeting_take2_mono.wav"
# FILE_PATH = "../resource/meeting_voice/meeting_demo_special_record_deluxe.wav"


# 照合用個人の音声
# download audioURL from web app. we necessary to convert to wav.

# FILE_PATH_DICT = {"hanagata": "./resource/record241023/sample_hanagata.wav", 
#                   "isizawa": "./resource/record241023/sample_ishizawa.wav",
#                   "shibasaki": "./resource/record241023/sample_shibasaki.wav"}
# FILE_PATH_DICT = {"hanagata": "../resource/onsei_kaigi1015/hanagata_sample_voice.wav", 
#                   "isizawa": "../resource/meeting_voice/ishizawa_sample.wav",
#                   "shibasaki": "../resource/onsei_kaigi1015/shibasaki_sample.wav"}


# 話者の情報をDjangoのmodelから取得

# wavファイルを保存するディレクトリ
output_dir = Path("sampleVoices")
output_dir.mkdir(exist_ok=True)

# すべてのオブジェクトを取得
speaker_models = speaker_model.objects.all()

data = [] # 辞書リストを格納するためのリスト

# Base64をデコードしてwavファイルに保存
for speaker_model in speaker_models:
    # Base64データのデコード
    wav_data = base64.b64decode(speaker_model.audio_base64)

    # 出力ファイル名の作成
    file_name = output_dir / f"{speaker_model.name}.wav"

    # wavファイルとして保存
    with open(file_name, "wb") as audio_file:
        audio_file.write(wav_data)

    # データをリストに追加
    data.append({
        "file_path" : str(file_name),
        "name" : speaker_model.name,
        "sex" : speaker_model.sex
    })

# データフレームに変換
speaker_df = pd.DataFrame(data)


# temp df
df = pd.DataFrame(
    [["./resource/record241023/sample_hanagata.wav","hanagata", "male"],
    ["./resource/record241023/sample_ishizawa.wav","isizawa", "male"],
    ["./resource/record241023/sample_shibasaki.wav","shibasaki", "female"]],
    columns=['file_path', 'name', 'sex']
)


ref = []
res_meet = model.transcribe(FILE_PATH, verbose=False, language="ja")
embed_meet = generate_speaker_embeddings(FILE_PATH,res_meet)

    
for id, row in df.iterrows():
    res = model.transcribe(row['file_path'], verbose=False, language="ja")
    embed = generate_speaker_embeddings(row['file_path'],res)
    ref.append((row['name'], row['sex'], embed))
close_ref_speaker_df = format_speaker_output_by_segment2(embed_meet,res_meet,ref)

print(close_ref_speaker_df)

# 出力結果
#                                                  text       name     sex
# 0   ではこれから新製品のマーケティング戦略に関する 社内会議を始めます まず新製品のターゲット層...   hanagata    male
# 1   でも最近の若い人たちも 同じ製品を求めている傾向があると 現場の声から聞いています それに価...    isizawa    male
# 2           いや30代以上に限定すべきだと思います 若い層はどうせ似たようなものを買うでしょう   hanagata    male
# 3      私たちが開発した技術は 幅広い年齢層にアピールできるものです 技術的にも他の層に響く可能性が  shibasaki  female
# 4           でもリソースが限られているんです 今のマーケティング戦略を変更する余裕はありません   hanagata    male
# 5                                         それは理解していますが    isizawa    male
# 6                                       もし市場の動向が変わったら   hanagata    male
# 7                              最初からシェアを狭めるのはリスクだと思います    isizawa    male
# 8            うーんそこまで考える時間はないんです 今はこの路線で行きます はい わかりました   hanagata    male
# 9                                   しかしもし失敗したらその時の責任は  shibasaki  female
# 10                          失敗しません 私たちはこれで行きます はい以上です   hanagata    male