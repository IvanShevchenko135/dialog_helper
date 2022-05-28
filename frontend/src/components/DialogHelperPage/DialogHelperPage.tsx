import React, { FC, useState } from 'react';
import { MicroButton } from '../MicroButton/MicroButton';
import { Header } from '../Header/Header';
import { PageState } from '../../types';
import './DialogHelperPage.scss';
import { Listener } from '../Listener/Listener';

export const DialogHelperPage: FC = () => {
    const [state, setState] = useState(PageState.INACTIVE);
    let content;

    switch (state) {
        case PageState.INACTIVE: {
            content = <MicroButton setPageState={setState} />;
            break;
        }

        case PageState.ACTIVE: {
            content = <Listener setPageState={setState} />;
            break;
        }
    }

    return (
        <div className="DialogHelperPage">
            <Header />
            <div className={`DialogHelperPage-Content DialogHelperPage-Content_state_${state}`}>
                {content}
            </div>
        </div>
    );
};
