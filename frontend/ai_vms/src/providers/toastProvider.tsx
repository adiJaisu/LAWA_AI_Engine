import React, { createContext, useContext, useRef, useMemo } from 'react';
import { Toast } from 'primereact/toast';
import type { ToastMessage, Toast as ToastRef } from 'primereact/toast';

type ToastContextValue = {
    showToast: (msg: ToastMessage | ToastMessage[]) => void;
    clearToast: () => void;
};

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export const ToastProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
    const toastRef = useRef<ToastRef>(null);

    const api = useMemo<ToastContextValue>(
        () => ({
            showToast: (msg) => toastRef.current?.show(msg),
            clearToast: () => toastRef.current?.clear(),
        }),
        []
    );

    return (
        <ToastContext.Provider value={api}>
            {/* Global Toast, rendered once and appended to body to avoid clipping */}
            <Toast ref={toastRef} position="top-right" appendTo={document.body} />
            {children}
        </ToastContext.Provider>
    );
};

export const useToast = () => {
    const ctx = useContext(ToastContext);
    if (!ctx) {
        throw new Error('useToast must be used within a ToastProvider');
    }
    return ctx;
};