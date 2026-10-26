import React from 'react';
import { BookOpen, Brain, Activity, TrendingUp, Clock, AlertCircle, CheckCircle } from 'lucide-react';

export const Methodology: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8 flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-blue-600" />
          About the Project
        </h1>

        {/* Problem Statement */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600" />
            The Industrial Sensor Missing-Data Problem
          </h2>
          <div className="prose text-gray-700">
            <p>
              Industrial equipment generates continuous sensor data streams. Due to sensor malfunctions, 
              communication errors, or maintenance activities, some observations may be missing. 
              This system reconstructs those missing values using both mathematical and neural approaches.
            </p>
            <p>
              <strong>Dataset:</strong> PHM 2018 Ion Mill Etch Tool dataset with 17 sensor features including 
              pressure, voltage, current, flow rates, and rotation parameters from semiconductor manufacturing equipment.
            </p>
            <p>
              <strong>Task Type:</strong> Offline reconstruction using both left and right context. 
              This is <em>not</em> real-time forecasting - we have observations before and after the missing section.
            </p>
          </div>
        </div>

        {/* Four Approaches with Expandable Details */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-600" />
            The Four Reconstruction Approaches
          </h2>

          {/* Linear Interpolation */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
            <h3 className="font-semibold text-gray-900 mb-2">1. Linear Interpolation</h3>
            <p className="text-sm text-gray-600 mb-3">
              Mathematical baseline that estimates missing values using linear interpolation between observed boundary values.
            </p>
            <details className="text-sm">
              <summary className="cursor-pointer text-blue-600 font-medium">How it works</summary>
              <div className="mt-2 text-gray-700">
                <p>
                  Estimates missing values by drawing a straight line between the last observed value before the gap 
                  and the first observed value after the gap.
                </p>
                <code className="block bg-white p-2 rounded mt-2">
                  prediction[t] = (1 - α) × last_observed + α × first_observed
                  <br />
                  where α = (t - gap_start) / (M + 1)
                </code>
                <p className="mt-2">
                  <strong>Characteristics:</strong> No training required, works for any gap length, very fast (near-instantaneous).
                </p>
              </div>
            </details>
          </div>

          {/* Attention Seq2Seq */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
            <h3 className="font-semibold text-gray-900 mb-2">2. Attention Seq2Seq</h3>
            <p className="text-sm text-gray-600 mb-3">
              Deep learning model with bidirectional LSTM encoder, additive attention, and LSTM decoder.
            </p>
            <details className="text-sm">
              <summary className="cursor-pointer text-blue-600 font-medium">How it works</summary>
              <div className="mt-2 text-gray-700">
                <p>
                  <strong>Architecture:</strong> Encoder-decoder sequence model with attention mechanism.
                </p>
                <ul className="list-disc pl-6 mt-2 space-y-1">
                  <li><strong>Encoder:</strong> Bidirectional LSTM (2 layers, 64 hidden units) processes left context</li>
                  <li><strong>Attention:</strong> Additive attention learns which context timesteps are most relevant</li>
                  <li><strong>Decoder:</strong> LSTM (2 layers, 64 hidden units) generates predictions step-by-step</li>
                </ul>
                <p className="mt-2">
                  <strong>Training:</strong> Trained for 50 epochs with early stopping. Best validation loss at epoch 48.
                  ~331K parameters. Only supports gap length 10.
                </p>
              </div>
            </details>
          </div>

          {/* Hybrid Reconstruction */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
            <h3 className="font-semibold text-gray-900 mb-2">3. Hybrid Reconstruction</h3>
            <p className="text-sm text-gray-600 mb-3">
              Combines linear interpolation and Attention Seq2Seq using a fixed-weight blend.
            </p>
            <details className="text-sm">
              <summary className="cursor-pointer text-blue-600 font-medium">How it works</summary>
              <div className="mt-2 text-gray-700">
                <code className="block bg-white p-2 rounded">
                  final_prediction = α × seq2seq_prediction + (1 - α) × interpolation_prediction
                  <br />
                  where α = 0.1 (10% neural, 90% interpolation)
                </code>
                <p className="mt-2">
                  <strong>Characteristics:</strong> Fusion weight determined from training data (not test data). 
                  Only supports gap length 10. Inference time is sum of both methods.
                </p>
              </div>
            </details>
          </div>

          {/* Residual LSTM */}
          <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
            <h3 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
              4. Residual LSTM
              <span className="text-xs bg-orange-200 text-orange-800 px-2 py-1 rounded">Preferred AI</span>
            </h3>
            <p className="text-sm text-gray-600 mb-3">
              Learns corrections to linear interpolation using a trained LSTM-based model.
            </p>
            <details className="text-sm">
              <summary className="cursor-pointer text-orange-600 font-medium">How it works</summary>
              <div className="mt-2 text-gray-700">
                <code className="block bg-white p-2 rounded">
                  final_prediction = interpolation_prediction + learned_correction
                </code>
                <p className="mt-2">
                  <strong>Architecture:</strong> LSTM encoder that takes left context, right context, and interpolation prediction.
                  The model predicts the residual (error) between interpolation and ground truth, then adds it back.
                </p>
                <p className="mt-2">
                  <strong>Training:</strong> Separate models trained for gap lengths 10, 20, and 50.
                  ~83K-127K parameters. Supports multiple gap lengths.
                </p>
              </div>
            </details>
          </div>
        </div>

        {/* Artificial Masking */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Artificial Masking for Evaluation
          </h2>
          <div className="prose text-gray-700">
            <p>
              To evaluate reconstruction methods, we artificially mask sections of the data where we actually 
              have ground-truth values. The process:
            </p>
            <ol className="list-decimal pl-6 space-y-2">
              <li>Take a complete sensor sequence with no missing values</li>
              <li>Define a window: N observations before, M observations to mask, N observations after</li>
              <li>Hide the middle M observations from the reconstruction method</li>
              <li>Let the method predict the hidden values</li>
              <li>Compare predictions against the actual (hidden) ground truth</li>
            </ol>
            <p>
              <strong>Important:</strong> Ground truth is never passed into the model as input for the missing region. 
              It is only used after prediction to calculate metrics and visualize prediction error.
            </p>
          </div>
        </div>

        {/* Epochs and Training */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-600" />
            Training Epochs and Loss Metrics
          </h2>
          <div className="prose text-gray-700">
            <p>
              <strong>What is an epoch?</strong> One complete pass through the entire training dataset.
              Training for multiple epochs allows the model to learn iteratively.
            </p>
            <p>
              <strong>Training Loss:</strong> Error on the training data. Should decrease during training 
              as the model learns to fit the training examples.
            </p>
            <p>
              <strong>Validation Loss:</strong> Error on a held-out validation set not used for training. 
              Used to select the best checkpoint and detect overfitting.
            </p>
            <p>
              <strong>Test Metrics:</strong> Final evaluation on a completely held-out test set. 
              These metrics (MSE, MAE, RMSE) are what we report as the model's performance.
            </p>
            <p>
              <strong>Early Stopping:</strong> Training stopped when validation loss stopped improving 
              (at epoch 48 for Attention Seq2Seq). This prevents overfitting.
            </p>
          </div>
        </div>

        {/* Gap Length Impact */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            Why Performance Changes with Gap Length
          </h2>
          <div className="prose text-gray-700">
            <p>
              <strong>Short gaps (e.g., 10 timesteps):</strong>
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Linear interpolation works very well - the signal doesn't change much in 10 steps</li>
              <li>Neural models have limited room for improvement</li>
              <li>Computational cost of neural methods may not be justified</li>
            </ul>
            <p>
              <strong>Longer gaps (e.g., 20, 50 timesteps):</strong>
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Linear interpolation may miss non-linear patterns over longer intervals</li>
              <li>Neural models have more opportunity to learn complex patterns</li>
              <li>Residual LSTM shows modest improvements for gaps 20 and 50</li>
              <li>Absolute error is higher for all methods (longer gaps are harder)</li>
            </ul>
          </div>
        </div>

        {/* Trade-offs */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Trade-offs: Error vs. Inference Time
          </h2>
          <div className="prose text-gray-700">
            <p>
              <strong>Linear Interpolation:</strong> Near-instantaneous, often best for short gaps.
            </p>
            <p>
              <strong>Attention Seq2Seq:</strong> ~10-50ms inference time, trained for gap 10 only.
            </p>
            <p>
              <strong>Hybrid Reconstruction:</strong> ~20-100ms (sum of both methods), gap 10 only.
            </p>
            <p>
              <strong>Residual LSTM:</strong> ~10-50ms inference time, supports gaps 10, 20, 50.
            </p>
            <p>
              <strong>Recommendation:</strong> Start with linear interpolation. Consider neural methods only when:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Gaps are longer (20+ timesteps)</li>
              <li>Small improvements matter for your application</li>
              <li>Computational budget allows the extra latency</li>
            </ul>
          </div>
        </div>

        {/* Project Conclusion */}
        <div className="bg-gray-100 rounded-lg shadow p-6 border-2 border-gray-300">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-gray-700" />
            Project Conclusion
          </h2>
          <div className="prose text-gray-700">
            <p className="text-base">
              Our experiments show that AI-based reconstruction does not automatically outperform mathematical interpolation. 
              Linear interpolation performed better for the short gap in our reported evaluation, while Residual LSTM achieved 
              modestly lower MSE for the longer gaps tested. We therefore select Residual LSTM as our preferred AI candidate 
              for further development, while retaining linear interpolation as a fast baseline and fallback. 
              Further evaluation is required before making an industrial deployment claim.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
