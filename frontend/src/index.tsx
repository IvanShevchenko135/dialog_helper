import React, { FC } from 'react';
import ReactDOM from 'react-dom/client';
import { DialogHelperPage } from './components/DialogHelperPage/DialogHelperPage';
import './index.scss';

const App: FC = () => {
    return (
        <DialogHelperPage />
    );
};

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement,
);

root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
);
