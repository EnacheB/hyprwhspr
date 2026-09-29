"""Experimental persistent-context Parakeet.cpp backend; native loading is lazy."""

import contextlib
import ctypes
import json
import os
import numpy as np

from .base import TranscriptionBackend, log
try:
    from .. import parakeet_cpp_runtime as runtime
except ImportError:
    import parakeet_cpp_runtime as runtime


def _shorten_pauses(audio, sample_rate, keep_seconds=1.0):
    """Shorten every pause longer than keep_seconds to keep_seconds.

    Parakeet silently drops whole sentences spoken after a pause of 1.5 s or
    more, even in 15 s recordings: 2 of 100 real dictations lost one, and this
    restored both without losing speech elsewhere. A 20 ms frame counts as
    silent when it is within 12 dB of the audio's own quietest 10%.
    """
    hop = int(0.02 * sample_rate)
    frames = audio.size // hop
    if not frames:
        return audio
    power = np.square(audio[:frames * hop], dtype=np.float64).reshape(frames, hop).mean(1)
    level = 10 * np.log10(power + 1e-12)
    silent = np.concatenate(([0], level < np.percentile(level, 10) + 12, [0])).astype(np.int8)
    edges = np.flatnonzero(np.diff(silent)) * hop
    keep = np.ones(audio.size, dtype=bool)
    half = int(keep_seconds / 2 * sample_rate)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > keep_seconds * sample_rate:
            keep[start + half:end - half] = False
    return audio if keep.all() else audio[keep]


class ParakeetCppBackend(TranscriptionBackend):
    name = 'parakeet-cpp'
    loads_in_background = True

    # Beam search rather than greedy: on real dictation it drops fillers and
    # the fragments greedy decoding invents from noise ("range vanquisher
    # title."). 4 wide costs ~0.2 s per 10 s of audio on a Radeon 780M.
    # parakeet.cpp v0.5.0's beam search throws on about 1 chunk in 5
    # ("zero-duration expansion did not reduce score"); the fix is
    # mudler/parakeet.cpp#74.
    _BEAM_SIZE = 4

    def __init__(self, manager):
        super().__init__(manager)
        self._library = None
        self._context = None
        self.device = None

    def initialize(self):
        # WhisperManager serializes initialize/transcribe/unload/cleanup with its
        # model lock. Never download or mutate process-wide environment here.
        if self.is_loaded:
            return True
        self.ready = False
        try:
            self.device = runtime.resolve_device(self.config, probe=False)
            path = runtime.library_path(self.device)
            if not path.is_file() or not runtime.model_installed():
                raise RuntimeError('runtime or pinned model missing')
            self._library = runtime.bind_library(path)
            self._context = self._library.parakeet_capi_load(os.fsencode(runtime.model_path()))
            if not self._context:
                raise RuntimeError('native model initialization failed (see native diagnostic)')
            self.current_model = runtime.MODEL_ID
            self.ready = True
            log(f'[PARAKEET-CPP] Ready: {runtime.RELEASE}, {self.device}, {runtime.MODEL_ID}')
            return True
        except Exception as exc:
            self.unload()
            log(f'[PARAKEET-CPP] Initialization failed: {exc}. Run hyprwhspr setup '
                '(Parakeet → Parakeet.cpp), select reinstall.')
            return False

    def transcribe(self, audio_data, sample_rate=16000, language_override=None):
        if not self.is_loaded:
            log('[PARAKEET-CPP] Model not loaded; run hyprwhspr setup and reinstall Parakeet.cpp')
            return ''
        output = None
        try:
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim != 1:
                raise ValueError('Expected mono audio')
            if not audio.size:
                return ''
            # parakeet.cpp resamples to 16 kHz itself, linearly. Keep that: judged
            # by ear on 100 real 44.1 kHz dictations, it beat resampling with soxr
            # first in 6 places and lost in 3.
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if audio.size > 2147483647 or not np.isfinite(audio).all():
                raise ValueError('Audio length or samples are invalid')
            audio = _shorten_pauses(audio, int(sample_rate))
            # Best hypothesis only, length-normalized like NeMo's beam search.
            output = self._library.parakeet_capi_transcribe_pcm_nbest_json(
                self._context, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                audio.size, int(sample_rate), self._BEAM_SIZE, 1, 1, None)
            if not output:
                error = self._library.parakeet_capi_last_error(self._context)
                raise RuntimeError((error or b'Native inference failed').decode('utf-8', errors='replace'))
            hypotheses = json.loads(ctypes.string_at(output).decode('utf-8'))['hypotheses']
            return hypotheses[0]['text'].strip() if hypotheses else ''
        except Exception as exc:
            log(f'[PARAKEET-CPP] Transcription failed: {exc}')
            return ''
        finally:
            if output:
                self._library.parakeet_capi_free_string(output)

    def _model_lock(self):
        # Resume recovery and shutdown call in without the manager's lock; freeing
        # the native context mid-inference is a use-after-free. RLock: re-entrant.
        return getattr(self._manager, '_model_lock', None) or contextlib.nullcontext()

    def unload(self):
        with self._model_lock():
            context, self._context = self._context, None
            self.ready = False
            if context:
                self._library.parakeet_capi_free(context)

    def reinitialize(self):
        with self._model_lock():
            self.unload()
            return self.initialize()

    def cleanup(self):
        self.unload()

    # Only a GPU context can go stale across suspend or a long idle; reloading
    # the CPU model would just add seconds to the next dictation.
    @property
    def reinit_on_idle(self):
        return self.device == 'vulkan'

    @property
    def reinit_on_resume(self):
        return self.device == 'vulkan'

    @property
    def is_loaded(self):
        return self._context is not None and bool(self._context)
