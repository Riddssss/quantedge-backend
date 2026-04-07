# transformer.py — Pure NumPy Transformer signal generator

import numpy as np


def positional_encoding(seq_len, d_model):
    PE  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    ang = pos / np.power(10000, (2 * (dim // 2)) / d_model)
    PE[:, 0::2] = np.sin(ang[:, 0::2])
    PE[:, 1::2] = np.cos(ang[:, 1::2])
    return PE


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def self_attention(x, W_Q, W_K, W_V):
    Q       = x @ W_Q
    K       = x @ W_K
    V       = x @ W_V
    scores  = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V


def get_signal(window, W_Q, W_K, W_V, W_out, b_out, PE):
    """Get one signal value for a single window"""
    x        = window + PE
    attn_out = self_attention(x, W_Q, W_K, W_V)
    pooled   = attn_out.mean(axis=0)
    return sigmoid(float(pooled @ W_out) + b_out)


def train(train_data, close_idx, n_features,
          seq_len=60, d_model=16, lr=0.05, epochs=8):
    """Train transformer weights on training data"""
    np.random.seed(42)
    sc    = np.sqrt(2.0 / n_features)
    W_Q   = np.random.randn(n_features, d_model) * sc
    W_K   = np.random.randn(n_features, d_model) * sc
    W_V   = np.random.randn(n_features, d_model) * sc
    W_out = np.random.randn(d_model) * np.sqrt(2.0 / d_model)
    b_out = 0.0
    PE    = positional_encoding(seq_len, n_features)

    for _ in range(epochs):
        indices = np.random.permutation(
            len(train_data) - seq_len)[:800]
        for i in indices:
            window = train_data[i : i + seq_len]
            label  = 1 if (
                train_data[i + seq_len, close_idx]
                > train_data[i + seq_len - 1, close_idx]) else 0
            pred   = get_signal(window, W_Q, W_K, W_V,
                                 W_out, b_out, PE)
            error  = pred - label
            x      = window + PE
            attn   = self_attention(x, W_Q, W_K, W_V)
            pooled = attn.mean(axis=0)
            grad   = error * pred * (1 - pred)
            W_out -= lr * grad * pooled
            b_out -= lr * grad

    return W_Q, W_K, W_V, W_out, b_out, PE


def signals_for_df(df_scaled, feature_cols,
                   W_Q, W_K, W_V, W_out, b_out, PE, seq_len):
    """Generate signals for entire dataframe"""
    data    = df_scaled[feature_cols].values
    signals = []
    for i in range(seq_len, len(data)):
        window = data[i - seq_len : i]
        signals.append(get_signal(window, W_Q, W_K,
                                   W_V, W_out, b_out, PE))
    pad = [0.5] * seq_len
    return np.array((pad + signals)[:len(df_scaled)])


def generate_transformer_signals(train_df, test_df,
                                  train_scaled, test_scaled,
                                  feature_cols, seq_len=60):
    """Main function — returns train_df and test_df with signal column"""
    data       = train_scaled[feature_cols].values
    close_idx  = feature_cols.index('Close')
    n_features = len(feature_cols)

    W_Q, W_K, W_V, W_out, b_out, PE = train(
        data, close_idx, n_features, seq_len=seq_len)

    train_sig = signals_for_df(train_scaled, feature_cols,
                                W_Q, W_K, W_V,
                                W_out, b_out, PE, seq_len)
    test_sig  = signals_for_df(test_scaled, feature_cols,
                                W_Q, W_K, W_V,
                                W_out, b_out, PE, seq_len)

    train_df = train_df.copy().reset_index(drop=True)
    test_df  = test_df.copy().reset_index(drop=True)

    train_df['Transformer_Signal'] = train_sig[:len(train_df)]
    test_df['Transformer_Signal']  = test_sig[:len(test_df)]

    return train_df, test_df