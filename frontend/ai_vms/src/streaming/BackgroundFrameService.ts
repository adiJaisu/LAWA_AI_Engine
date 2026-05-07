import { http } from "../api/axiosInstance";

export type GetTokenFn = () => string | null;

export interface FrameServiceOptions {
    intervalMs?: number;
    width?: number;
    height?: number;
    mimeType?: "image/jpeg" | "image/webp";
    quality?: number;
    videoConstraints?: MediaTrackConstraints;
    fieldName?: string;
    extraFields?: Record<string, string>;
}

export function createBackgroundFrameService(
    options?: FrameServiceOptions
) {
    const intervalMs = options?.intervalMs ?? 333;
    const captureWidth = options?.width ?? 1920;
    const captureHeight = options?.height ?? 1080;
    const mimeType = options?.mimeType ?? "image/jpeg";
    const quality = Math.min(1, Math.max(0.2, options?.quality ?? 0.7));
    const fieldName = options?.fieldName ?? "frame";
    const extraFields = options?.extraFields ?? {};
    const videoConstraints: MediaTrackConstraints = options?.videoConstraints ?? {
        width: { ideal: 640 },
        height: { ideal: 360 },
        frameRate: { ideal: 10 },
    };

    // ---- internal state ----
    let stream: MediaStream | null = null;
    let video: HTMLVideoElement | null = null;
    let canvas: HTMLCanvasElement | null = null;
    let ctx: CanvasRenderingContext2D | null = null;
    let timer: number | null = null;
    let starting = false;

    async function start() {
    if (timer !== null || starting) return;
    starting = true;

    try {

         // 2) Acquire camera
        const userData = sessionStorage.getItem("user");
        if (!userData) {
            console.warn("User not found");
            return;
        }
        const user = JSON.parse(userData);
        if (user.role_id !== 1) {
            console.log("Camera disabled for this role");
            return;
        }
        stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false,
        });

        // 3) Hidden <video>
        video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;
        await video.play();

        // 4) Offscreen canvas
        canvas = document.createElement("canvas");
        canvas.width = captureWidth;
        canvas.height = captureHeight;
        ctx = canvas.getContext("2d", { alpha: false });

        // 5) Fire first capture immediately, then set interval
        await captureAndUpload();
        timer = window.setInterval(() => {
            captureAndUpload().catch((err) => {
                // Log but don’t crash loop
                console.warn("[frame-service] upload error:", err);
            });
        }, intervalMs) as unknown as number;
    } catch (err) {
        console.error("[frame-service] start failed:", err);
        // On error, release anything half-initialized
        await stop();
    } finally {
        starting = false;
    }
}

    async function stop() {
        if (timer !== null) {
            clearInterval(timer);
            timer = null;
        }
        if (video) {
            try { video.pause(); video.srcObject = null; } catch { }
            video = null;
        }
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        ctx = null;
        canvas = null;
    }

    async function captureAndUpload() {
        if (!video || !canvas || !ctx) return;

        // Draw current frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Encode to chosen MIME (JPEG by default). If you want WebP, set mimeType to "image/webp".
        const blob: Blob | null = await new Promise((resolve) =>
            canvas!.toBlob(resolve, mimeType, quality)
        );

        if (!blob || blob.size === 0) {
            // If this happens, browser doesn't support the MIME; try fallback to JPEG
            if (mimeType !== "image/jpeg") {
                const jpegFallback = await new Promise<Blob | null>((resolve) =>
                    canvas!.toBlob(resolve, "image/jpeg", quality)
                );
                if (!jpegFallback) return;
                await sendBlob(jpegFallback, "image/jpeg");
            }
            return;
        }

        await sendBlob(blob, mimeType);
    }

    async function sendBlob(b: Blob, effectiveMime: string) {
        const filename =
            effectiveMime === "image/webp"
                ? `frame-${Date.now()}.webp`
                : `frame-${Date.now()}.jpg`;
        const cameraJson = sessionStorage.getItem("camera");
        if (cameraJson === null) {
            console.error('Camera Id Not Found');
            return;
        }
        const cameraId = JSON.parse(cameraJson);
        const file = new File([b], filename, { type: effectiveMime });
        const form = new FormData();
        form.append(fieldName, file);
        form.append("format", effectiveMime);
        form.append("ts", String(Date.now()));
        form.append("width", String(canvas!.width));
        form.append("height", String(canvas!.height));
        form.append("cameraId", cameraId.camera_id);


        for (const [k, v] of Object.entries(extraFields)) {
            form.append(k, v);
        }
        const res = await http.post(
            `/streaming/start`,
            form
        );
        if (!res.status) {
            const text = await res.data.text().catch(() => "");
            console.warn("[frame-service] upload failed:", res.status, text);
        }
    }

    return { start, stop };
}