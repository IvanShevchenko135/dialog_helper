export interface Prediction {
    word: string,
    translation: string,
}

export enum PageState {
    INACTIVE = 'inactive',
    ACTIVE = 'active',
}
