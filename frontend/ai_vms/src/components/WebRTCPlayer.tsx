import React, { useEffect, useRef, useState } from 'react';

interface WebRTCPlayerProps {
    cameraId: string | number;
    className?: string;
    onResolutionUpdate?: (resolution: string) => void;
}

const WebRTCPlayer: React.FC<WebRTCPlayerProps> = ({ cameraId, className, onResolutionUpdate }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const pcRef = useRef<RTCPeerConnection | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [connecting, setConnecting] = useState(true);

    useEffect(() => {
        let isMounted = true;

        async function startWebRTC() {
            try {
                setConnecting(true);
                setError(null);

                const pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                pcRef.current = pc;

                pc.addTransceiver('video', { direction: 'recvonly' });

                pc.ontrack = (event) => {
                    if (videoRef.current && event.streams[0]) {
                        videoRef.current.srcObject = event.streams[0];
                    }
                };

                pc.onconnectionstatechange = () => {
                    if (!isMounted) return;
                    console.log(`[WebRTC|${cameraId}] Connection state: ${pc.connectionState}`);
                    if (pc.connectionState === 'connected') setConnecting(false);
                    if (pc.connectionState === 'failed') setError('Connection failed');
                };

                // Create Offer
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                // Send to Backend Proxy
                const host = window.location.hostname;
                const response = await fetch(`http://${host}:8010/api/v1/stream/webrtc/offer`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription?.sdp,
                        type: pc.localDescription?.type,
                        camera_id: String(cameraId)
                    })
                });

                if (!response.ok) {
                    throw new Error(`Signaling failed: ${response.statusText}`);
                }

                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));

                if (isMounted && answer.resolution && onResolutionUpdate) {
                    onResolutionUpdate(answer.resolution);
                }

            } catch (err: any) {
                if (isMounted) {
                    console.error(`[WebRTC|${cameraId}] Error:`, err);
                    setError(err.message);
                    setConnecting(false);
                }
            }
        }

        startWebRTC();

        return () => {
            isMounted = false;
            if (pcRef.current) {
                pcRef.current.close();
                pcRef.current = null;
            }
        };
    }, [cameraId]);

    return (
        <div className={`webrtc-player-container ${className}`} style={{ width: '100%', height: '100%', position: 'relative', backgroundColor: '#000' }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
            {connecting && !error && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '12px', background: 'rgba(0,0,0,0.5)' }}>
                    Connecting WebRTC...
                </div>
            )}
            {error && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ff4444', fontSize: '12px', textAlign: 'center', padding: '10px' }}>
                    {error}
                </div>
            )}
        </div>
    );
};

export default WebRTCPlayer;
