use burn::tensor::{
    activation::relu, activation::softmax, backend::Backend, BasicOps, Element, ElementConversion,
    Numeric, Tensor, TensorKind,
};


use std::f32::NEG_INFINITY;
use std::vec;
use std::result;

use strum_macros::EnumIter;

use num_traits::ToPrimitive;

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const WINDOW_LENGTH: usize = N_FFT;
const LANGUAGES: [&str; 1]=[
    "en"
];

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;
    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let qk = q.matmul(k);

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    //normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);

    return o;
}

pub fn attn_decoder_mask<B: Backend>(seq_length: usize) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length]);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (i + 1)]).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }
    return mask;
}

// Helper functions
pub struct Helper {}

impl Helper {
    pub fn tensor_max_scalar<B: Backend, const D: usize>(
        x: Tensor<B, D>,
        max: f64,
    ) -> Tensor<B, D> {
        relu(x.sub_scalar(max)).add_scalar(max)
    }

    pub fn tensor_min_scalar<B: Backend, const D: usize>(
        x: Tensor<B, D>,
        min: f64,
    ) -> Tensor<B, D> {
        -Self::tensor_max_scalar(-x, -min)
    }

    pub fn tensor_max<B: Backend, const D: usize>(
        x: Tensor<B, D>,
        max: Tensor<B, D>,
    ) -> Tensor<B, D> {
        relu(x - max.clone()) + max
    }

    pub fn tensor_min<B: Backend, const D: usize>(
        x: Tensor<B, D>,
        min: Tensor<B, D>,
    ) -> Tensor<B, D> {
        -Self::tensor_max(-x, -min)
    }

    pub fn tensor_log10<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        let ln10 = (10.0f64).ln();
        x.log() / ln10
    }

    pub fn all_zeros<B: Backend, const D: usize>(x: Tensor<B, D>) -> bool {
        x.abs().max().into_scalar().to_f64().unwrap() == 0.0
    }

    pub fn _10pow<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        let log10 = (10.0f64).ln();
        (x * log10).exp()
    }

    pub fn reverse<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B> + Numeric<B>>(
        x: Tensor<B, D, K>,
        dim: usize,
    ) -> Tensor<B, D, K>
    where
        <K as BasicOps<B>>::Elem: Element,
    {
        let len = x.dims()[dim];
        let indices = -Tensor::arange_device(0..len, &x.device()) + (len - 1) as i64;
        x.select(dim, indices)
    }
}

pub struct Audio {}

impl Audio {
    /// Returns the maximum number of waveform samples that can be sumbitted to `prep_audio` without
    /// receiving more than `n_frame_max` frames.
    pub fn max_waveform_samples(n_frame_max: usize) -> usize {
        // The number of waveform samples must be less than this
        let n_samples_max = HOP_LENGTH * (n_frame_max + 1) + Self::is_odd(N_FFT) as usize;
        n_samples_max - 1
    }

    fn is_odd(x: usize) -> bool {
        if x % 2 == 0 {
            false
        } else {
            true
        }
    }

    pub fn fft_frequencies_device<B: Backend>(
        sample_rate: f64,
        n_fft: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        Tensor::arange_device(0..(n_fft / 2 + 1), device)
            .float()
            .mul_scalar(sample_rate / n_fft as f64)
    }
    pub fn hz_to_mel(freq: f64, htk: bool) -> f64 {
        if htk {
            return 2595.0 * (1.0 + freq / 700.0).log10();
        }

        // Fill in the linear part
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;

        // File in the log-scale part

        let min_log_hz = 1000.0; // beginning of log region (Hz)
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f64).ln() / 27.0; // step size for log region

        let mel = if freq >= min_log_hz {
            min_log_mel + (freq / min_log_hz).ln() / logstep
        } else {
            (freq - f_min) / f_sp
        };

        return mel;
    }

    pub fn mel_to_hz_tensor<B: Backend>(mel: Tensor<B, 1>, htk: bool) -> Tensor<B, 1> {
        if htk {
            return (Helper::_10pow(mel / 2595.0) - 1.0) * 700.0;
        }

        // Fill in the linear scale
        let f_min = 0.0;
        let f_sp = 200.0 / 3.0;

        // And now the nonlinear scale
        let min_log_hz = 1000.0; // beginning of log region (Hz)
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f64).ln() / 27.0; // step size for log region

        let log_t = mel.clone().greater_equal_elem(min_log_mel).float();
        let freq = log_t.clone() * (((mel.clone() - min_log_mel) * logstep).exp() * min_log_hz)
            + (-log_t + 1.0) * (mel * f_sp + f_min);

        return freq;
    }

    pub fn mel_frequencies_device<B: Backend>(
        n_mels: usize,
        fmin: f64,
        fmax: f64,
        htk: bool,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        // `Center freqs` of mel bands -uniformly spaced between limits
        let min_mel = Self::hz_to_mel(fmin, htk);
        let max_mel = Self::hz_to_mel(fmax, htk);

        //mels=np.linspace(min_mel, max_mel, n_mels)
        let mels = Tensor::arange_device(0..n_mels, device)
            .float()
            .mul_scalar((max_mel - min_mel) / (n_mels - 1) as f64)
            .add_scalar(min_mel);

        //hz: np.ndarray = mel_to_hz(mels, htk=htk)
        Self::mel_to_hz_tensor(mels, htk)
    }

    pub fn get_mel_filters_device<B: Backend>(
        sample_rate: f64,
        n_fft: usize,
        n_mels: usize,
        htk: bool,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let fmin = 0.0;
        let fmax = sample_rate * 0.5;

        // Center freqs of each FFT bin
        let fftfreqs = Self::fft_frequencies_device(sample_rate, n_fft, device);
        let [n_ffefreqs] = fftfreqs.dims();

        // `Center freqs` of the mel bands - uniformly spaced between limits
        let mel_f_size = n_mels + 2;
        let mel_f = Self::mel_frequencies_device(mel_f_size, fmin, fmax, htk, device);

        // fdiff=np.diff(mel_f)
        let fdiff =
            mel_f.clone().slice([1..mel_f_size]) - mel_f.clone().slice([0..(mel_f_size - 1)]);

        // ramps=np.subtract.outer(mel_f, fftfreqs)
        let ramps = mel_f
            .clone()
            .unsqueeze::<2>()
            .transpose()
            .repeat(1, n_ffefreqs)
            - fftfreqs.unsqueeze();

        // lower and upper slopes for all bins
        let lower = -ramps.clone().slice([0..n_mels])
            / fdiff
                .clone()
                .slice([0..n_mels])
                .unsqueeze::<2>()
                .transpose();

        let upper = ramps.slice([2..(2 + n_mels)])
            / fdiff.slice([1..(1 + n_mels)]).unsqueeze::<2>().transpose();

        // .. then intersect them with each other and zero
        let weights = relu(Helper::tensor_min(lower, upper));

        // Slaney-style mel is scaled to be approx constant energy per channel
        // enorm=2.0/(mel_f[2:n_mels+2]-mel_f[:n_mels])
        let enorm = (mel_f.clone().slice([2..(n_mels + 2)]) - mel_f.clone().slice([0..n_mels]))
            .powf(-1.0)
            * 2.0;

        //weights *=enorm[:, np.newaxis]
        let weights = weights * enorm.unsqueeze::<2>().transpose();

        if !(Helper::all_zeros(mel_f.slice([0..(n_mels - 2)]))
            || Helper::all_zeros(relu(-weights.clone().max_dim(1))))
        {
            println!("Empty filters detected in mel frequency basis. \nSome channels will produce empty responses. \nTry increasing your sampling rate (and fmax) or reducing n_mels.");
        }
        return weights;
    }

    /// Transform an input waveform into a format interpretable by Whisper.
    /// With a waveform size of (n_batch, n_samples) the output will be of size (n_batch, n_mels, n_frame)
    /// where n_mels = 80
    /// n_frame=int((n_samples_padded-n_fft)/hop_length),
    /// n_samples_padded=if n_fft is even: n_samples+n_fft else: n_samples +n_fft-1,
    /// n_fft=400.
    /// hop_length=160.
    pub fn pre_audio<B: Backend>(waveform: Tensor<B, 2>, sample_rate: f64) -> Tensor<B, 3> {
        let device = waveform.device();
        let window = Self::hann_window_device(WINDOW_LENGTH, &device);
        let (stft_real, stft_imag) = Self::stfft(waveform, N_FFT, HOP_LENGTH, window);

        let magnitudes = stft_real.powf(2.0) + stft_imag.powf(2.0);
        let [n_batch, n_row, n_col] = magnitudes.dims();
        let magnitudes = magnitudes.slice([0..n_batch, 0..n_row, 0..(n_col - 1)]);

        let mel_spec = Self::get_mel_filters_device(sample_rate, N_FFT, N_MELS, false, &device)
            .unsqueeze()
            .matmul(magnitudes);

        let log_spec = Helper::tensor_log10(Helper::tensor_max_scalar(mel_spec, 1.0e-10));

        let max: f64 = log_spec.clone().max().into_scalar().elem();

        let log_spec = Helper::tensor_max_scalar(log_spec, max - 8.0);
        let log_spec = (log_spec + 4.0) / 4.0;

        return log_spec;
    }

    pub fn hann_window<B: Backend>(window_length: usize) -> Tensor<B, 1> {
        Self::hann_window_device(window_length, &B::Device::default())
    }

    pub fn hann_window_device<B: Backend>(
        window_length: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        Tensor::arange_device(0..window_length, device)
            .float()
            .mul_scalar(std::f64::consts::PI / window_length as f64)
            .sin()
            .powf(2.0)
    }

    pub fn div_roundup(a: usize, b: usize) -> usize {
        (a + b - 1) / b
    }

    /// short time fourier transform that takes a waveform input of size (n_batch, n_sample) and returns (real_part, imaginary_part) frequency spectrums.
    /// The size of each returned tensor is (n_batch, n_freq, n_frame) when n_freq =int(n_fft/2+1), n_frame=int((n_sample_padded-n_fft)/hop_length)+1,
    /// n_sample_padded=if n_fft is even: n_sample+n_fft else: n_sample+n_fft-1.
    pub fn stfft<B: Backend>(
        input: Tensor<B, 2>,
        n_fft: usize,
        hop_length: usize,
        window: Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [n_batch, orig_input_size] = input.dims();
        assert!(orig_input_size >= n_fft);

        let device = input.device();

        // add reflection padding to center the windows on the input times
        let pad = n_fft / 2;
        let left_pad = Helper::reverse(input.clone().slice([0..n_batch, 1..(pad + 1)]), 1);
        let right_pad = Helper::reverse(
            input.clone().slice([
                0..n_batch,
                (orig_input_size - pad - 1)..(orig_input_size - 1),
            ]),
            1,
        );
        let input = Tensor::cat(vec![left_pad, input, right_pad], 1);

        // pad window to length n_fft
        let [orig_window_length] = window.dims();
        let window = if orig_window_length < n_fft {
            let left_pad = (n_fft - orig_window_length) / 2;
            let right_pad = n_fft - orig_window_length - left_pad;

            Tensor::cat(
                vec![
                    Tensor::zeros_device([left_pad], &device),
                    window,
                    Tensor::zeros_device([right_pad], &device),
                ],
                0,
            )
        } else {
            window
        };

        let [_, input_size] = input.dims();

        let n_frame = (input_size - n_fft) / hop_length + 1;
        // assuming real input there is conjugate symmetry
        let n_feq = n_fft / 2 + 1;

        // construct matrix of overlapping input windows
        let num_parts = Self::div_roundup(n_fft, hop_length);
        let n_hops = Self::div_roundup(input_size, hop_length);
        let padded_input_size = n_hops * hop_length;
        let padding = Tensor::zeros_device([n_batch, padded_input_size - input_size], &device);
        let template = Tensor::cat(vec![input, padding], 1)
            .reshape([n_batch, n_hops, hop_length])
            .transpose();

        let parts: Vec<_> = (0..num_parts)
            .into_iter()
            .map(|i| {
                template
                    .clone()
                    .slice([0..n_batch, 0..hop_length, i..(n_frame + i)])
            })
            .collect();

        let input_windows = Tensor::cat(parts, 1).slice([0..n_batch, 0..n_fft, 0..n_frame]);

        // construct matrix of wave angles
        let coe = std::f64::consts::PI * 2.0 / n_fft as f64;
        let b = Tensor::arange_device(0..n_feq, &device)
            .float()
            .mul_scalar(coe)
            .unsqueeze::<2>()
            .transpose()
            .repeat(1, n_fft)
            * Tensor::arange_device(0..n_fft, &device)
                .float()
                .unsqueeze::<2>();

        // convolve the input slices with the window and waves
        let real_part = (b.clone().cos() * window.clone().unsqueeze())
            .unsqueeze()
            .matmul(input_windows.clone());
        let imaginary_part = (b.sin() * (-window).unsqueeze())
            .unsqueeze()
            .matmul(input_windows);
        return (real_part, imaginary_part);
    }
}

#[derive(Clone)]
pub struct BeamNode<T: Clone> {
    pub seq: Vec<T>,
    pub log_prob: f64,
}

pub struct Beam {}

impl Beam {
    pub fn get_top_elements<T>(elems: &[T], score: impl Fn(&T) -> f64, num: usize) -> Vec<&T> {
        let mut top_elems = Vec::with_capacity(num);
        let mut scores = Vec::with_capacity(num);

        for elem in elems {
            let score = score(elem);

            if top_elems.len() == num {
                if score < scores[0] {
                    continue;
                }
            }

            if let Some((idx, _)) = scores.iter().enumerate().find(|(_, &s)| s >= score) {
                top_elems.insert(idx, elem);
                scores.insert(idx, score);
            } else {
                top_elems.push(elem);
                scores.push(score);
            }

            if top_elems.len() > num {
                top_elems.remove(0);
                scores.remove(0);
            }
        }

        top_elems
    }

    pub fn beam_search<T, F, G>(
        initial_beams: Vec<BeamNode<T>>,
        next: F,
        is_finished: G,
        beam_size: usize,
        max_depth: usize,
    ) -> Vec<T>
    where
        T: Clone,
        F: Fn(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>> + Clone,
        G: Fn(&[T]) -> bool + Clone,
    {
        let mut beams = initial_beams;
        for i in 0..max_depth {
            if let Some(beam) = beams
                .iter()
                .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap())
            {
                if is_finished(&beam.seq) {
                    break;
                }
            }
            beams = Self::beam_search_step(beams, next.clone(), is_finished.clone(), beam_size);
            print!("Depath: {} ", i);
        }

        beams
            .into_iter()
            .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap())
            .map(|x| x.seq)
            .unwrap_or_else(Vec::new)
    }

    pub fn beam_search_step<T, F, G>(
        beams: Vec<BeamNode<T>>,
        next: F,
        is_finished: G,
        beam_size: usize,
    ) -> Vec<BeamNode<T>>
    where
        T: Clone,
        F: Fn(&[BeamNode<T>]) -> Vec<Vec<(T, f64)>>,
        G: Fn(&[T]) -> bool,
    {
        let mut finished_beams = Vec::with_capacity(beam_size);
        let mut new_beams = Vec::with_capacity(beam_size);
        let continuations = next(&beams);

        for (beam_node, continuations) in beams.into_iter().zip(continuations) {
            if is_finished(&beam_node.seq) {
                finished_beams.push(beam_node);
            } else {
                let top_new_beams =
                    Self::get_top_elements(&continuations, |(_, log_prob)| *log_prob, beam_size)
                        .into_iter()
                        .map(move |(tok, log_prob)| BeamNode {
                            seq: [beam_node.seq.clone(), vec![tok.clone()]].concat(),
                            log_prob: *log_prob,
                        });
                new_beams.extend(top_new_beams);
            }
        }

        Self::get_top_elements(&new_beams, |beam| beam.log_prob, beam_size)
            .into_iter()
            .chain(Self::get_top_elements(
                &finished_beams,
                |beam| beam.log_prob,
                beam_size,
            ))
            .cloned()
            .collect()
    }
}




#[derive(Debug, Copy, Clone, EnumIter)]
pub enum Language {
    English
}

impl Language{
    pub fn as_str(&self)-> &str{
        match self {
            Language::English=> "en",
        }
    }
}

pub enum SpecialToken {
    EndofText,
    StartofTranscript,
    Translate,
    Transcribe,
    StartofLM,
    StartoPrev,
    NoSpeech,
    NoTimeStamps,
    Language(Language),
    Timestamp(f64),
}

impl ToString for SpecialToken{
    fn to_string(&self) -> String {
        match self {
            SpecialToken::EndofText => "<|endoftext|>".into(),
            SpecialToken::StartofTranscript => "<|startoftranscript|>".into(),
            SpecialToken::Translate => "<|translate|>".into(),
            SpecialToken::Transcribe => "<|transcribe|>".into(),
            SpecialToken::StartofLM => "<|startoflm|>".into(),
            SpecialToken::StartoPrev => "<|startofprev|>".into(),
            SpecialToken::NoSpeech => "<|nospeech|>".into(),
            SpecialToken::NoTimeStamps => "<|notimestamps|>".into(),
            SpecialToken::Language(lang) => format!("<|{}|>", lang.as_str()),
            SpecialToken::Timestamp(val) => format!("<|{:.2}|>", val),
        }
    }
}