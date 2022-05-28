import React, { FC } from 'react';
import { Icon } from '../Icon/Icon';
import './Header.scss';

export const Header: FC = () => {
    return (
        <div className="DialogHelperPage-Header Header">
            <div className="Header-AppName">{'Dialog Helper'}</div>
            <div className="Header-Settings">
                <Icon className="Header-SettingsIcon" type="settings" />
            </div>
        </div>
    );
};
