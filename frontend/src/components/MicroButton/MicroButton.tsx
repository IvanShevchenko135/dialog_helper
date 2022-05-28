import React, { FC, useCallback } from 'react';
import { Icon } from '../Icon/Icon';
import { PageState } from '../../types';
import './MicroButton.scss';

interface MicroButtonProps {
    setPageState: (state: PageState) => void;
}

export const MicroButton: FC<MicroButtonProps> = props => {
    const { setPageState } = props;

    const startApp = useCallback(() => {
        fetch('/api/start').catch(console.error);
        setPageState(PageState.ACTIVE);
    }, []);

    return (
        <div className="MicroButton" onClick={startApp}>
            <Icon className="MicroButton-Icon" type="microphone" />
        </div>
    );
};
