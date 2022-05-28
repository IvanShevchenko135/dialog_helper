import React, { FC, RefObject, useEffect, useRef, useState } from 'react';
import { PageState, Prediction } from '../../types';
import './Listener.scss';

interface ListenerProps {
    setPageState: (state: PageState) => void;
    barsNumber?: number;
}

const dafaultPredictions = (length: number) => new Array(length).fill(null).map(() => {
    return {
        word: '-',
        translation: '',
    };
});

const getAudioAnalyser = () => {
    const audioCtx = new AudioContext();
    return navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(stream => {
        const analyser = audioCtx.createAnalyser();
        analyser.smoothingTimeConstant = 0.7;
        analyser.fftSize = 32;

        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        return analyser;
    });
};

export const Listener: FC<ListenerProps> = props => {
    const { barsNumber = 16 } = props;

    const visualizerRef: RefObject<HTMLDivElement> = useRef(null);
    const frequencyData = new Uint8Array(barsNumber);

    const [intervalId, setIntervalId] = useState<NodeJS.Timer | null>(null);

    const [transcript, setTranscript] = useState<string>('');
    const [predictions, setPredictions] = useState<Prediction[]>(dafaultPredictions(3));

    const initListener = () => {
        if (visualizerRef.current && !intervalId) {
            const soundBars = visualizerRef.current.querySelectorAll<HTMLDivElement>('.Listener-SoundBar');

            const id = setInterval(() => {
                fetch('/api/transcript').then(res => res.json()).then(data => {
                    data.transcript && setTranscript(data.transcript);
                });
            }, 100);

            setInterval(() => {
                fetch('/api/prediction').then(res => res.json()).then(data => {
                    data.predictions && setPredictions(data.predictions);
                });
            }, 100);

            setIntervalId(id);

            getAudioAnalyser().then(audioAnalyser => {
                const renderFrame = () => {
                    audioAnalyser.getByteFrequencyData(frequencyData);
                    Object.values(frequencyData).forEach((value, i) => {
                        const valuePercent = Math.max(value / 255, 0.1);
                        const bar = soundBars[i];

                        bar.style.transform = `scaleY(${valuePercent})`;
                        bar.style.opacity = `${valuePercent}`;
                    });

                    requestAnimationFrame(renderFrame);
                };
                requestAnimationFrame(renderFrame);
            });
        }
    };

    useEffect(initListener, []);

    return (
        <div>
            <div className="Listener-VisualizerContainer">
                <div className="Listener-SpeakLabel">Говорите</div>
                <div ref={visualizerRef} className="Listener-Visualizer">
                    {Array(barsNumber).fill(null).map((_, i) => {
                        return <div key={i} className="Listener-SoundBar" />;
                    })}
                </div>
            </div>
            <div className="Listener-Transcript">
                {transcript}
            </div>
            <div className="Listener-Predictions">
                {predictions.map((prediction, i) => {
                    return (
                        <div key={i} className="Listener-Prediction">
                            <div className="Listener-PredictionWord">{prediction.word}</div>
                            <div className="Listener-PredictionTranslation">{prediction.translation}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
