import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn, fft


class FFTEngine(nn.Module):
    fft_algo = {
        'fft': (fft.fft, fft.ifft),
        'rfft': (fft.rfft, fft.irfft),
        'hfft': (fft.hfft, fft.ihfft)
    }

    def __init__(self, algo):
        super().__init__()
        self.fft, self.ifft = self.fft_algo[algo]

    def forward(self, x):
        _sp = self.fft(x)
        xrc = self.ifft(_sp)
        return _sp, xrc


def get_input(input_file=None, n=101, s=20, sym=True) -> (torch.Tensor, int):
    if input_file is None:
        offset = (n - 1) / 2 if sym else 0
        x_ = (np.arange(0, n) - offset) / s
        x2_ = np.cos(x_) * np.cos(2 * x_)
        ml = x2_.size
        x2_ = torch.tensor(x2_)
        return x2_, ml
    else:
        df = pd.read_csv(input_file, header=None, names=['probe', 'intensity'])
        x2_ = df['intensity'].values
        ml = x2_.size
        truncate_signal = st.sidebar.number_input(f"Truncate Signal Min", value=1e-6, step=1e-6)
        t = (x2_ > truncate_signal).nonzero()[0]

        dp_range = st.sidebar.slider("data points range", value=[int(t[0]), int(t[-1])])
        st.sidebar.markdown(f"data points: {dp_range[1]-dp_range[0]}/{t[-1]-t[0]}/{ml}")
        st.sidebar.markdown("数据点数量影响基波空间频率`K0`")
        st.sidebar.markdown("数据点间隔影响基波空间频率`K0`")
        return torch.tensor(x2_[slice(*dp_range)]), ml


class AppV1:
    def exec(self):
        input_file_options = [f"data/{2 ** x}cm-1.csv" for x in [0, 1, 2, 3, 4, 5]][::-1]

        input_file_ = st.sidebar.selectbox('input_file', options=[*input_file_options, None])
        algo = st.sidebar.selectbox('algorithm', options=['rfft', 'fft', 'hfft'])

        if input_file_ is None:
            n_ = st.sidebar.number_input('n', min_value=20, max_value=10000, value=101, step=1)
            s_ = st.sidebar.number_input('s', min_value=2, max_value=500, value=20, step=1)
            sym_ = st.sidebar.checkbox('symmetric signal', value=True)

        else:
            n_ = 0
            s_ = 0
            sym_ = True

        ffte = FFTEngine(algo)
        ffte_module = torch.jit.script(ffte)
        print(ffte_module)
        x_input, max_length = get_input(input_file=input_file_, n=n_, s=s_, sym=sym_)

        a, b = ffte_module(torch.tensor(x_input))

        st.markdown("# Welcome to FFT App ")
        st.markdown("## Raw input ")
        st.bar_chart(pd.DataFrame(x_input.numpy()))
        st.markdown("## FFT results ")
        sp = np.abs(a.numpy())
        spectrum_limit = st.sidebar.slider(
            'spectrum',
            value=[0, int(max_length/2)])
        st.sidebar.write(f"FFT length: {spectrum_limit[1] - spectrum_limit[0]}/{len(sp)}")
        st.bar_chart(pd.DataFrame(sp[slice(*spectrum_limit)]))

        st.markdown("## Reconstructed signals ")
        st.bar_chart(pd.DataFrame(b.resolve_conj().numpy().astype('float')))


AppV1().exec()
