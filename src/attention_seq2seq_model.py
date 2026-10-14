import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    Encoder outputs:
        [batch, context_length, 2 * hidden_dim]

    Decoder hidden state:
        [batch, hidden_dim]

    Returns:
        context vector [batch, 2 * hidden_dim]
        attention weights [batch, context_length]
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [B, T, E]
        # decoder_hidden:  [B, H]

        encoder_features = self.encoder_projection(encoder_outputs)
        decoder_features = self.decoder_projection(
            decoder_hidden
        ).unsqueeze(1)

        energy = torch.tanh(
            encoder_features + decoder_features
        )

        scores = self.score(energy).squeeze(-1)

        attention_weights = F.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class AttentionSeq2SeqReconstructor(nn.Module):
    """
    Attention-based bidirectional Seq2Seq model for reconstructing
    a masked sensor segment.

    Input:
        context_before: [B, context_length, input_dim]
        context_after:  [B, context_length, input_dim]

    Output:
        reconstructed segment: [B, output_length, output_dim]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=None,
        output_length=10,
        dropout=0.3,
        attention_dim=128,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_length = output_length

        encoder_dim = hidden_dim * 2

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # ---------------------------------------------------------
        # Bridge encoder -> decoder
        # ---------------------------------------------------------

        self.hidden_bridge = nn.ModuleList([
            nn.Linear(encoder_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.cell_bridge = nn.ModuleList([
            nn.Linear(encoder_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # ---------------------------------------------------------
        # Attention
        # ---------------------------------------------------------

        self.attention = AdditiveAttention(
            encoder_dim=encoder_dim,
            decoder_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        # ---------------------------------------------------------
        # Timestep embedding
        # ---------------------------------------------------------

        self.timestep_embedding = nn.Embedding(
            output_length,
            hidden_dim,
        )

        # ---------------------------------------------------------
        # Previous prediction embedding
        # ---------------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        decoder_input_dim = (
            hidden_dim
            + encoder_dim
            + hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ---------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------

        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_dim + encoder_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # ---------------------------------------------------------
        # Learned start token
        # ---------------------------------------------------------

        self.start_token = nn.Parameter(
            torch.zeros(1, output_dim)
        )

        nn.init.normal_(
            self.start_token,
            mean=0.0,
            std=0.02,
        )

    def encode(self, context_before, context_after):
        """
        Encode both sides of the missing region.

        The two context windows are concatenated in temporal order:

            before -> after

        Shape:
            [B, 50, 17] + [B, 50, 17]
            ->
            [B, 100, 17]
        """

        context = torch.cat(
            [context_before, context_after],
            dim=1,
        )

        encoder_outputs, (hidden, cell) = self.encoder(context)

        # hidden/cell:
        # [num_layers * 2, B, H]

        batch_size = context.size(0)

        hidden = hidden.view(
            self.num_layers,
            2,
            batch_size,
            self.hidden_dim,
        )

        cell = cell.view(
            self.num_layers,
            2,
            batch_size,
            self.hidden_dim,
        )

        decoder_hidden = []
        decoder_cell = []

        for layer in range(self.num_layers):

            # Concatenate forward and backward states.
            h = torch.cat(
                [
                    hidden[layer, 0],
                    hidden[layer, 1],
                ],
                dim=1,
            )

            c = torch.cat(
                [
                    cell[layer, 0],
                    cell[layer, 1],
                ],
                dim=1,
            )

            h = torch.tanh(
                self.hidden_bridge[layer](h)
            )

            c = torch.tanh(
                self.cell_bridge[layer](c)
            )

            decoder_hidden.append(h)
            decoder_cell.append(c)

        decoder_hidden = torch.stack(
            decoder_hidden,
            dim=0,
        )

        decoder_cell = torch.stack(
            decoder_cell,
            dim=0,
        )

        return (
            encoder_outputs,
            decoder_hidden,
            decoder_cell,
        )

    def decode_step(
        self,
        previous_output,
        timestep,
        encoder_outputs,
        hidden,
        cell,
    ):
        """
        Perform one autoregressive decoder step.
        """

        # Previous predicted/target sensor vector.
        previous_embedding = self.input_projection(
            previous_output
        )

        # Timestep embedding.
        timestep_ids = torch.full(
            (previous_output.size(0),),
            timestep,
            dtype=torch.long,
            device=previous_output.device,
        )

        timestep_embedding = self.timestep_embedding(
            timestep_ids
        )

        # Current top decoder hidden state.
        decoder_state = hidden[-1]

        # Attention over all encoder positions.
        context, attention_weights = self.attention(
            encoder_outputs,
            decoder_state,
        )

        decoder_input = torch.cat(
            [
                previous_embedding,
                context,
                timestep_embedding,
            ],
            dim=1,
        ).unsqueeze(1)

        decoder_output, (hidden, cell) = self.decoder(
            decoder_input,
            (hidden, cell),
        )

        decoder_output = decoder_output.squeeze(1)

        combined = torch.cat(
            [
                decoder_output,
                context,
            ],
            dim=1,
        )

        prediction = self.output_projection(
            combined
        )

        return (
            prediction,
            hidden,
            cell,
            attention_weights,
        )

    def forward(
        self,
        context_before,
        context_after,
        target=None,
        teacher_forcing_ratio=0.5,
        return_attention=False,
    ):
        """
        Forward pass.

        During training:
            target can be supplied and teacher forcing is used.

        During evaluation:
            target=None and predictions are autoregressive.
        """

        (
            encoder_outputs,
            hidden,
            cell,
        ) = self.encode(
            context_before,
            context_after,
        )

        batch_size = context_before.size(0)

        previous_output = self.start_token.expand(
            batch_size,
            -1,
        )

        predictions = []
        attention_history = []

        for timestep in range(self.output_length):

            (
                prediction,
                hidden,
                cell,
                attention_weights,
            ) = self.decode_step(
                previous_output=previous_output,
                timestep=timestep,
                encoder_outputs=encoder_outputs,
                hidden=hidden,
                cell=cell,
            )

            predictions.append(prediction.unsqueeze(1))

            if return_attention:
                attention_history.append(
                    attention_weights.unsqueeze(1)
                )

            # -----------------------------------------------------
            # Real teacher forcing
            # -----------------------------------------------------

            if (
                target is not None
                and timestep < self.output_length - 1
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                previous_output = target[:, timestep, :]
            else:
                previous_output = prediction

        predictions = torch.cat(
            predictions,
            dim=1,
        )

        if return_attention:
            attention_history = torch.cat(
                attention_history,
                dim=1,
            )

            return predictions, attention_history

        return predictions


if __name__ == "__main__":

    print("Testing AttentionSeq2SeqReconstructor...")

    batch_size = 4
    context_length = 50
    gap_length = 10
    sensor_count = 17

    model = AttentionSeq2SeqReconstructor(
        input_dim=sensor_count,
        hidden_dim=64,
        num_layers=2,
        output_dim=sensor_count,
        output_length=gap_length,
        dropout=0.3,
        attention_dim=128,
    )

    context_before = torch.randn(
        batch_size,
        context_length,
        sensor_count,
    )

    context_after = torch.randn(
        batch_size,
        context_length,
        sensor_count,
    )

    target = torch.randn(
        batch_size,
        gap_length,
        sensor_count,
    )

    # Training-style forward pass.
    output = model(
        context_before,
        context_after,
        target=target,
        teacher_forcing_ratio=0.5,
    )

    print(
        "Training output shape:",
        tuple(output.shape),
    )

    # Inference-style forward pass.
    with torch.no_grad():
        prediction = model(
            context_before,
            context_after,
            target=None,
            teacher_forcing_ratio=0.0,
        )

    print(
        "Inference output shape:",
        tuple(prediction.shape),
    )

    # Attention inspection.
    with torch.no_grad():
        prediction, attention = model(
            context_before,
            context_after,
            target=None,
            teacher_forcing_ratio=0.0,
            return_attention=True,
        )

    print(
        "Attention shape:",
        tuple(attention.shape),
    )

    print(
        "Expected prediction shape:",
        (batch_size, gap_length, sensor_count),
    )

    print(
        "Expected attention shape:",
        (batch_size, gap_length, context_length * 2),
    )

    assert output.shape == (
        batch_size,
        gap_length,
        sensor_count,
    )

    assert prediction.shape == (
        batch_size,
        gap_length,
        sensor_count,
    )

    assert attention.shape == (
        batch_size,
        gap_length,
        context_length * 2,
    )

    print()
    print("MODEL TEST PASSED")
