class ExperimentModel(BaseModel):
    """
    This class allows to set all parameters individually.
    It should only be used when you want to try out some custom architectures.
    """
    def __init__(self, model_id: str, input_shape: tuple[int, int]):
        super().__init__(model_id=model_id, input_shape=input_shape)

    def build(self, vocab_sizes: dict[str, int], sequence_length: int = 32):
        input_layers = {
            'bar': Input(shape=(sequence_length,), name='input_bar'),
            'position': Input(shape=(sequence_length,), name='input_position'),
            'pitch': Input(shape=(sequence_length,), name='input_pitch'),
            'duration': Input(shape=(sequence_length,), name='input_duration'),
            'velocity': Input(shape=(sequence_length,), name='input_velocity'),
            'tempo': Input(shape=(sequence_length,), name='input_tempo')
        }

        embedding_layers = {
            'bar': Embedding(input_dim=vocab_sizes['bar'], output_dim=8, name='embedding_bar')
                (input_layers['bar']),
            'position': Embedding(input_dim=vocab_sizes['position'], output_dim=16, name='embedding_position')
                (input_layers['position']),
            'pitch': Embedding(input_dim=vocab_sizes['pitch'], output_dim=48, name='pitch_embedding_pitch')
                (input_layers['pitch']),
            'duration': Embedding(input_dim=vocab_sizes['duration'], output_dim=16, name='embedding_duration')
                (input_layers['duration']),
            'velocity': Embedding(input_dim=vocab_sizes['velocity'], output_dim=16, name='embedding_velocity')
                (input_layers['velocity']),
            'tempo': Embedding(input_dim=vocab_sizes['tempo'], output_dim=8, name='embedding_tempo')
                (input_layers['tempo'])
        }

        x = Concatenate(name="concat_all_embeddings")(list(embedding_layers.values()))

        x = LSTM(units=128, return_sequences=True, name="lstm_1", recurrent_dropout=0.075)(x)
        x = Dropout(rate=0.1)(x)
        x = LSTM(units=64, return_sequences=False, name="lstm_2", recurrent_dropout=0.075)(x)
        x = Dropout(rate=0.15)(x)


        output_layers = {
            'bar': Dense(units=vocab_sizes['bar'], activation='softmax', name='output_bar')(x),
            'position': Dense(units=vocab_sizes['position'], activation='softmax', name='output_position')(x),
            'pitch': Dense(units=vocab_sizes['pitch'], activation='softmax', name='output_pitch')(x),
            'duration': Dense(units=vocab_sizes['duration'], activation='softmax', name='output_duration')(x),
            'velocity': Dense(units=vocab_sizes['velocity'], activation='softmax', name='output_velocity')(x),
            'tempo': Dense(units=vocab_sizes['tempo'], activation='softmax', name='output_tempo')(x)
        }

        model = Model(
            inputs=list(input_layers.values()),
            outputs=list(output_layers.values()),
            name=f"{self.model_id}_midi_lstm"
        )

        # The keys in this dict are not the keys o the output dict, but the layer names.
        loss_functions = {
            'output_bar': 'sparse_categorical_crossentropy',
            'output_position': 'sparse_categorical_crossentropy',
            'output_pitch': 'sparse_categorical_crossentropy',
            'output_duration': 'sparse_categorical_crossentropy',
            'output_velocity': 'sparse_categorical_crossentropy',
            'output_tempo': 'sparse_categorical_crossentropy'
        }

        loss_weights = {
            'output_bar': 0.2,
            'output_position': 0.6,
            'output_pitch': 0.8,
            'output_duration': 0.4,
            'output_velocity': 0.6,
            'output_tempo': 0.2
        }

        # Also the layer names as keys
        metric_functions = {
            'output_bar': 'accuracy',
            'output_position': 'accuracy',
            'output_pitch': 'accuracy',
            'output_duration': 'accuracy',
            'output_velocity': 'accuracy',
            'output_tempo': 'accuracy'
        }



        optimizer = Adam(learning_rate=1e-3)

        model.compile(
            optimizer=optimizer,
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=metric_functions
        )

        self.model = model
