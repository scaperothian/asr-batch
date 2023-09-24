import os
import time
import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

import numpy as np

torch.random.manual_seed(0)
device = torch.device('cpu')

print(f"PyTorch Version: {torch.__version__}, Pytorchaudio Version: {torchaudio.__version__}, Targeted Device: {device}")

def truncate_waveform(data: torch.Tensor,sr: float, seconds: float) ->torch.Tensor:
    """
    Summary: truncates tensor to length that cooresponds to sr * seconds
    
    Input Arguments: 
    data (torch.Tensor) - audio data
    sr (torch.Float) - sampling rate of audio data
    seconds - seconds of data to return. 
            if the seconds is specified to be larger than the len(tensor) / sr, then an error is raised.  
            if seconds is 0, then error is raised.
    
    Return: torch.Tensor that is truncated from input, or unmodified.
    
    """
    # calculate the truncation length of tensor.
    truncation_len = int(np.floor(sr * seconds))
    
    # pull out dimensions.
    dim, max_len = data.shape
    
    if max_len < truncation_len: 
        raise Exception(f"Expected seconds to be less than: {max_len / sr:.2f}.  Seconds is: {seconds}.")
    
    # check for errors in the input.
    if seconds == 0.0: 
        raise Exception("seconds cannot be 0.0.")

    #if max_len > truncation_len:
    #    print("normal truncation occuring.") 
    
    return data[:,:truncation_len]

if __name__ == "__main__":
    # Loading ASR Model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    SPEECH_FILE = "thinking_out_loud.wav"
    actual_transcript = "When your legs don't work like they used to before And I can't sweep you off of your feet Will your mouth still remember the taste of my love Will your eyes still smile from your cheeks And darling I will be loving you til we are seventy And baby my heart could still fall as hard at twenty three And I'm thinking about how people fall in love in mysterious ways Maybe just the touch of a hand Oh me I fall in love with you every single day And I just wanna tell you I am So honey now Take me into your loving arms Kiss me under the light of a thousand stars Place your head on my beating heart I'm thinking out loud Maybe we found love right where we are When my hair's all but gone and my memory fades And the crowds don't"
    actual_transcript = actual_transcript.split()
    max_length_secs = 120
    if os.path.exists(SPEECH_FILE):
        waveform, sample_rate = torchaudio.load(SPEECH_FILE)
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate
        waveform_trunc = truncate_waveform(waveform, sample_rate, seconds=max_length_secs)
        waveform = waveform_trunc.to(device)
        print(f"Target Sample Rate: {sample_rate}")
    else:
        print('NO FILE HERE!')
    dim, num_samples = waveform.size()
    print(f"Song is {num_samples / sample_rate:.2f} seconds long.  {num_samples} samples long.")


    files = download_pretrained_files("librispeech-4-gram")
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26
    
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        nbest=3,
        beam_size=1500,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    start = time.time()
    with torch.inference_mode():
        emission, _ = model(waveform)
        beam_search_result = beam_search_decoder(emission)
    
    finish = time.time()
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_result[0][0].words) / len(
        actual_transcript
    )
    
    print(f"Transcript: {beam_search_transcript}")
    print(f"Time to perform inference (with decoding): {finish-start:.1f} seconds.")
    print(f"Song is {num_samples / sample_rate:.2f} seconds long.  {num_samples} samples long.")
    print(f"WER: {beam_search_wer: .3f}")