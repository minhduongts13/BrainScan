// UploadPredictor.jsx


import React, { useState, useRef, useCallback } from 'react'

export default function UploadPredictor() {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const inputRef = useRef(null)

    const onFile = useCallback((f) => {
        if (!f) return
        setFile(f)
        setResult(null)
        setError(null)
        const url = URL.createObjectURL(f)
        setPreview(url)
    }, [])

    const onFileChange = (e) => onFile(e.target.files?.[0])

    const onDrop = (e) => {
        e.preventDefault()
        if (e.dataTransfer?.files?.[0]) onFile(e.dataTransfer.files[0])
    }

    const onDragOver = (e) => e.preventDefault()

    const reset = () => {
        setFile(null)
        setPreview(null)
        setResult(null)
        setError(null)
        if (inputRef.current) inputRef.current.value = ''
    }

    const onSubmit = async () => {
        if (!file) return setError('Vui lòng chọn ảnh trước')
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const fd = new FormData()
            fd.append('image', file)
            const res = await fetch('/api/predict', { method: 'POST', body: fd })
            if (!res.ok) throw new Error(await res.text())
            const data = await res.json()
            // expected: { label: 'hemorrhage'|'normal', prob: 0.92 }
            setResult(data)
        } catch (err) {
            setError(err.message || String(err))
        } finally {
            setLoading(false)
        }
    }

    const confidencePercent = result ? Math.round((result.prob || 0) * 100) : 0

    return (
        <div className="max-w-4xl mx-auto p-6">
            <header className="flex items-center gap-4 mb-6">
                <div className="p-3 rounded-xl bg-gradient-to-br from-sky-600 to-indigo-600 text-white shadow-lg">
                    {/* brain icon */}
                    <i className="fa-solid fa-brain text-3xl"></i>
                </div>

                <div>
                <h1 className="text-2xl font-semibold">Brain Hemorrhage Detector</h1>
                <p className="text-sm text-black-500">Upload a CT/MRI slice — model: MobileNet-based classifier</p>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left column: controls */}
                <div className="lg:col-span-1 bg-white border rounded-lg p-4 shadow-sm">
                <div
                    className="border-2 border-dashed border-gray-200 rounded-md p-4 text-center cursor-pointer hover:border-sky-300 transition"
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    onClick={() => inputRef.current?.click()}
                >
                    <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />

                    <div className="flex flex-col items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-12 h-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M3 15a4 4 0 004 4h10a4 4 0 100-8 5 5 0 00-9.9 1" />
                        <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M16 10l-4-4-4 4" />
                    </svg>
                    <p className="text-sm font-medium">Kéo & thả ảnh vào đây, hoặc bấm để chọn</p>
                    <p className="text-xs text-gray-400">Hỗ trợ JPG/PNG. Nên crop lấy slice não.</p>
                    </div>
                </div>

                <div className="mt-4 flex gap-2">
                    <button
                    onClick={onSubmit}
                    disabled={loading || !file}
                    className="flex-1 px-4 py-2 rounded bg-sky-600 text-white disabled:opacity-60"
                    >
                    {loading ? (
                        <span className="inline-flex items-center gap-2">
                        <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                        </svg>
                        Đang dự đoán...
                        </span>
                    ) : (
                        'Dự đoán'
                    )}
                    </button>

                    <button onClick={reset} className="px-4 py-2 rounded border">Reset</button>
                </div>

                {error && <p className="mt-3 text-sm text-red-600">Lỗi: {error}</p>}

                <div className="mt-4 text-xs text-gray-500">
                    <p><span className="font-medium">Ghi chú:</span> Model trả về label và xác suất. Kết quả chỉ để tham khảo, không thay thế chẩn đoán y tế.</p>
                </div>

                <div className="mt-4">
                    <p className="text-sm font-medium mb-2">Ví dụ (click để dùng)</p>
                    <div className="flex gap-2">
                    {/* Example thumbnails (replace urls) */}
                    <button className="w-16 h-12 bg-gray-100 rounded overflow-hidden border" onClick={() => onFile(new File([], ''))} disabled>
                        {/* placeholder */}
                    </button>
                    <button className="w-16 h-12 bg-gray-100 rounded overflow-hidden border" disabled />
                    </div>
                </div>
                </div>

                {/* Right column: preview + result */}
                <div className="lg:col-span-2 space-y-4">
                <div className="bg-white border rounded-lg p-3 shadow-sm h-96 flex items-center justify-center overflow-hidden relative">
                    {!preview && (
                    <div className="text-center text-gray-400">
                        <p className="font-medium">Chưa có ảnh</p>
                        <p className="text-sm">Chọn ảnh ở bên trái hoặc kéo thả vào khung</p>
                    </div>
                    )}

                    {preview && (
                    <>
                        <img src={preview} alt="preview" className="object-contain max-h-full max-w-full" />

                        {/* result badge */}
                        {result && (
                        <div className="absolute left-4 top-4 flex items-center gap-3">
                            <div className={`px-3 py-1 rounded-full text-sm font-semibold ${result.label === 'hemorrhage' ? 'bg-red-600 text-white' : 'bg-emerald-600 text-white'}`}>
                            {result.label === 'hemorrhage' ? 'Hemorrhage' : 'No hemorrhage'}
                            </div>

                            <div className="text-xs text-gray-200 bg-black/40 px-2 py-1 rounded">{(result.prob*100).toFixed(1)}%</div>
                        </div>
                        )}
                    </>
                    )}
                </div>

                {/* Diagnostic panel */}
                <div className="bg-white border rounded-lg p-4 shadow-sm">
                    <h3 className="font-medium mb-3">Diagnostics</h3>

                    {result ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                        <p className="text-sm text-gray-600 mb-2">Confidence</p>
                        <div className="w-full bg-gray-100 rounded h-4 overflow-hidden">
                            <div className={`${confidencePercent >= 70 ? 'bg-red-500' : 'bg-sky-500'} h-4`} style={{ width: `${confidencePercent}%`, transition: 'width 600ms' }} />
                        </div>
                        <p className="mt-2 text-sm font-semibold">{confidencePercent}%</p>
                        </div>

                        <div>
                        <p className="text-sm text-gray-600 mb-2">Model</p>
                        <div className="text-sm">MobileNet (fine-tuned)</div>
                        <p className="mt-3 text-xs text-gray-500">Notes: ensure input is a single axial slice for best accuracy.</p>
                        </div>
                    </div>
                    ) : (
                    <p className="text-sm text-gray-500">Kết quả sẽ hiển thị ở đây khi bạn thực hiện dự đoán.</p>
                    )}
                </div>
                </div>
            </div>

            <footer className="mt-6 text-xs text-gray-400 text-center">
                <p>Không sử dụng kết quả này để thay thế chẩn đoán y tế. Chỉ dùng cho mục đích nghiên cứu/học tập.</p>
            </footer>
        </div>
    )
}


