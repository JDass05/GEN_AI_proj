import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Search, 
  UploadCloud, 
  BarChart3, 
  Scale, 
  FileText, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  Layers, 
  Sparkles,
  ArrowRight,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Cpu
} from 'lucide-react';

const PRELOADED_CLAIMS = [
  "0-dimensional biomaterials show inductive properties.",
  "Epigenetic regulation is associated with sporadic cancers.",
  "ALDH1 expression is associated with better prognosis in breast cancer.",
  "Decreased expression of CXCR4 is associated with disease-free survival in breast cancer.",
  "MicroRNA-155 is upregulated in certain cancers."
];

export default function App() {
  const [activeTab, setActiveTab] = useState('checker'); // 'checker' | 'upload' | 'benchmark'
  const [claim, setClaim] = useState('');
  const [rounds, setRounds] = useState(2);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  
  // File upload state
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Benchmark state
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);

  // System status
  const [systemHealth, setSystemHealth] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setSystemHealth(data))
      .catch(() => {});
  }, []);

  const handleFactCheck = async (e) => {
    e?.preventDefault();
    if (!claim.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch('/api/fact-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim, rounds, top_k: topK })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Fact-checking request failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append('file', uploadFile);

    try {
      const res = await fetch('/api/upload-document', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      const data = await res.json();
      setUploadStatus({
        type: 'success',
        message: `Successfully ingested "${data.title}"! Added ${data.chunks_added} chunks to ChromaDB & BM25.`
      });
      setUploadFile(null);
    } catch (err) {
      setUploadStatus({
        type: 'error',
        message: err.message
      });
    } finally {
      setUploading(false);
    }
  };

  const loadBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch('/api/benchmark');
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'benchmark') {
      loadBenchmark();
    }
  }, [activeTab]);

  return (
    <div style={{ maxWidth: '1240px', margin: '0 auto', padding: '24px 20px' }}>
      
      {/* Header */}
      <header style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        paddingBottom: '20px',
        borderBottom: '1px solid var(--border-color)',
        marginBottom: '28px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ 
            background: 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)', 
            padding: '10px', 
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Scale size={28} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '22px', fontWeight: '800', letterSpacing: '-0.5px' }}>
              Consensus<span style={{ color: 'var(--accent-cyan)' }}>AI</span>
            </h1>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              Adversarial Multi-Agent Fact-Checking grounded in Hybrid RAG
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ 
          display: 'flex', 
          background: 'var(--bg-secondary)', 
          padding: '4px', 
          borderRadius: '10px',
          border: '1px solid var(--border-color)'
        }}>
          <button 
            onClick={() => setActiveTab('checker')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              background: activeTab === 'checker' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'checker' ? '#fff' : 'var(--text-muted)',
              transition: 'all 0.2s'
            }}
          >
            <Search size={15} /> Fact Checker
          </button>
          <button 
            onClick={() => setActiveTab('upload')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              background: activeTab === 'upload' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'upload' ? '#fff' : 'var(--text-muted)',
              transition: 'all 0.2s'
            }}
          >
            <UploadCloud size={15} /> Upload Knowledge
          </button>
          <button 
            onClick={() => setActiveTab('benchmark')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              background: activeTab === 'benchmark' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'benchmark' ? '#fff' : 'var(--text-muted)',
              transition: 'all 0.2s'
            }}
          >
            <BarChart3 size={15} /> Benchmark
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === 'checker' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
          
          {/* Input Panel */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <form onSubmit={handleFactCheck}>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', marginBottom: '8px', color: 'var(--text-muted)' }}>
                  NATURAL LANGUAGE CLAIM
                </label>
                <textarea 
                  rows={3}
                  value={claim}
                  onChange={(e) => setClaim(e.target.value)}
                  placeholder="Enter a scientific or factual claim to rigorously stress-test..."
                  style={{
                    width: '100%',
                    background: 'rgba(10, 13, 20, 0.7)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '10px',
                    padding: '12px 14px',
                    color: '#fff',
                    fontSize: '15px',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                />
              </div>

              {/* Sample Claims Pills */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-dim)', alignSelf: 'center' }}>Examples:</span>
                {PRELOADED_CLAIMS.map((ex, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setClaim(ex)}
                    style={{
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '16px',
                      padding: '4px 12px',
                      color: 'var(--text-muted)',
                      fontSize: '12px',
                      cursor: 'pointer',
                      textAlign: 'left'
                    }}
                  >
                    {ex.length > 45 ? ex.substring(0, 45) + '...' : ex}
                  </button>
                ))}
              </div>

              {/* Configuration Controls & Submit */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
                <div style={{ display: 'flex', gap: '20px' }}>
                  <div>
                    <label style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                      Debate Rounds: <b>{rounds}</b>
                    </label>
                    <input 
                      type="range" 
                      min="1" 
                      max="3" 
                      value={rounds}
                      onChange={(e) => setRounds(Number(e.target.value))}
                      style={{ width: '120px' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                      Top-K Chunks: <b>{topK}</b>
                    </label>
                    <input 
                      type="range" 
                      min="3" 
                      max="8" 
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      style={{ width: '120px' }}
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading || !claim.trim()}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '12px 28px',
                    borderRadius: '10px',
                    border: 'none',
                    background: loading ? 'var(--text-dim)' : 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                    color: '#fff',
                    fontWeight: '700',
                    fontSize: '14px',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    boxShadow: loading ? 'none' : '0 4px 15px rgba(99, 102, 241, 0.35)'
                  }}
                >
                  {loading ? (
                    <>Debating Evidence...</>
                  ) : (
                    <>Stress-Test Claim <ArrowRight size={16} /></>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Error Banner */}
          {error && (
            <div style={{
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              borderRadius: '10px',
              padding: '16px',
              color: '#fda4af',
              display: 'flex',
              gap: '12px',
              alignItems: 'center'
            }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {/* Result Output */}
          {result && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '20px' }}>
              
              {/* Verdict Summary Card */}
              <div className="glass-panel" style={{ padding: '24px', borderLeft: `6px solid ${
                result.verdict?.verdict === 'SUPPORTED' ? 'var(--accent-emerald)' :
                result.verdict?.verdict === 'REFUTED' ? 'var(--accent-rose)' : 'var(--accent-amber)'
              }` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    {result.verdict?.verdict === 'SUPPORTED' && <CheckCircle2 size={26} color="var(--accent-emerald)" />}
                    {result.verdict?.verdict === 'REFUTED' && <XCircle size={26} color="var(--accent-rose)" />}
                    {result.verdict?.verdict === 'NOT_ENOUGH_INFO' && <AlertCircle size={26} color="var(--accent-amber)" />}
                    <h2 style={{ fontSize: '20px', fontWeight: '800' }}>
                      VERDICT: {result.verdict?.verdict || 'UNKNOWN'}
                    </h2>
                  </div>
                  <div style={{ display: 'flex', gap: '14px' }}>
                    <span style={{ 
                      background: 'rgba(255,255,255,0.06)', 
                      padding: '6px 12px', 
                      borderRadius: '8px', 
                      fontSize: '13px' 
                    }}>
                      Confidence: <b>{Math.round((result.verdict?.confidence || 0) * 100)}%</b>
                    </span>
                    <span style={{ 
                      background: 'rgba(255,255,255,0.06)', 
                      padding: '6px 12px', 
                      borderRadius: '8px', 
                      fontSize: '13px' 
                    }}>
                      Citation Validity: <b>{Math.round(result.citation_validity_rate * 100)}%</b>
                    </span>
                  </div>
                </div>

                <p style={{ color: 'var(--text-main)', fontSize: '15px', lineHeight: '1.6', marginBottom: '16px' }}>
                  {result.verdict?.reasoning}
                </p>

                {/* Detected Fallacies */}
                {result.verdict?.detected_fallacies?.length > 0 && (
                  <div style={{ 
                    background: 'rgba(245, 158, 11, 0.08)', 
                    border: '1px solid rgba(245, 158, 11, 0.2)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    marginTop: '12px'
                  }}>
                    <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-amber)' }}>
                      FLAGGED RHETORICAL FALLACIES:
                    </span>
                    <ul style={{ margin: '6px 0 0 18px', fontSize: '13px', color: '#fde68a' }}>
                      {result.verdict.detected_fallacies.map((fal, i) => (
                        <li key={i}>{fal}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Debate Transcript */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Scale size={18} color="var(--accent-indigo)" /> Adversarial Debate Transcript ({result.transcript?.length} turns)
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {result.transcript?.map((turn, idx) => {
                    const isProponent = turn.role === 'PROPONENT';
                    return (
                      <div 
                        key={idx}
                        style={{
                          background: isProponent ? 'rgba(99, 102, 241, 0.08)' : 'rgba(244, 63, 94, 0.08)',
                          border: `1px solid ${isProponent ? 'rgba(99, 102, 241, 0.25)' : 'rgba(244, 63, 94, 0.25)'}`,
                          borderRadius: '12px',
                          padding: '16px'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <span style={{ 
                            fontSize: '12px', 
                            fontWeight: '800', 
                            color: isProponent ? '#818cf8' : '#fb7185' 
                          }}>
                            {turn.role} — ROUND {turn.round_num}
                          </span>
                          {turn.cited_chunk_ids?.length > 0 && (
                            <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                              Cited: {turn.cited_chunk_ids.join(', ')}
                            </span>
                          )}
                        </div>
                        <p style={{ fontSize: '14px', lineHeight: '1.6', color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>
                          {turn.argument}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Retrieved Chunks Accordion */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={18} color="var(--accent-cyan)" /> Top Grounding Evidence ({result.evidence?.length} chunks)
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {result.evidence?.map((chunk, idx) => (
                    <div 
                      key={idx}
                      style={{
                        background: 'rgba(10, 13, 20, 0.6)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '10px',
                        padding: '14px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-cyan)' }}>
                          [{chunk.chunk_id}] {chunk.title}
                        </span>
                        <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                          rerank score: {chunk.rerank_score?.toFixed(4)}
                        </span>
                      </div>
                      <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                        {chunk.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}

        </div>
      )}

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div style={{ maxWidth: '680px', margin: '0 auto' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px' }}>
              Add New Knowledge Files to RAG
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
              Upload scientific papers, notes, or articles (PDF, DOCX, TXT, MD). Chunks will be embedded onto ChromaDB and indexed into BM25 in real-time, making them immediately searchable in the adversarial debate.
            </p>

            <form onSubmit={handleFileUpload}>
              <div style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '14px',
                padding: '36px 20px',
                textAlign: 'center',
                background: 'rgba(10, 13, 20, 0.4)',
                cursor: 'pointer',
                marginBottom: '20px'
              }}
              onClick={() => document.getElementById('fileInput').click()}
              >
                <input 
                  id="fileInput"
                  type="file" 
                  accept=".pdf,.docx,.doc,.txt,.md"
                  style={{ display: 'none' }}
                  onChange={(e) => setUploadFile(e.target.files[0])}
                />
                <UploadCloud size={38} color="var(--accent-cyan)" style={{ marginBottom: '10px' }} />
                <p style={{ fontSize: '14px', fontWeight: '600' }}>
                  {uploadFile ? uploadFile.name : 'Click or drag file to upload'}
                </p>
                <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                  Supports PDF, DOCX, Markdown, Text
                </span>
              </div>

              <button
                type="submit"
                disabled={uploading || !uploadFile}
                style={{
                  width: '100%',
                  padding: '12px',
                  borderRadius: '10px',
                  border: 'none',
                  background: uploading ? 'var(--text-dim)' : 'var(--accent-cyan)',
                  color: '#000',
                  fontWeight: '700',
                  fontSize: '14px',
                  cursor: uploading || !uploadFile ? 'not-allowed' : 'pointer'
                }}
              >
                {uploading ? 'Chunking & Embedding on GPU...' : 'Index Document into RAG'}
              </button>
            </form>

            {uploadStatus && (
              <div style={{
                marginTop: '20px',
                padding: '14px',
                borderRadius: '10px',
                background: uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                border: `1px solid ${uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                color: uploadStatus.type === 'success' ? '#6ee7b7' : '#fda4af',
                fontSize: '13px'
              }}>
                {uploadStatus.message}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Benchmark Dashboard Tab */}
      {activeTab === 'benchmark' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '16px' }}>
              Multi-Agent Debate vs. Single-LLM Baseline Benchmark
            </h2>

            {benchmarkLoading && <p style={{ color: 'var(--text-muted)' }}>Loading evaluation metrics...</p>}

            {benchmarkData?.status === 'not_run' && (
              <div style={{ padding: '20px', background: 'rgba(255,255,255,0.03)', borderRadius: '10px' }}>
                <p style={{ color: 'var(--text-muted)', marginBottom: '10px' }}>{benchmarkData.message}</p>
                <code style={{ fontSize: '12px', color: 'var(--accent-cyan)' }}>
                  python -m src.evaluate --limit 10
                </code>
              </div>
            )}

            {benchmarkData?.metrics && (
              <div>
                {/* Metric Summary Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>ACCURACY GAIN</span>
                    <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '4px' }}>
                      {((benchmarkData.metrics.debate_accuracy || 0) * 100).toFixed(1)}%
                      <span style={{ fontSize: '13px', color: 'var(--accent-emerald)', marginLeft: '8px' }}>
                        vs {((benchmarkData.metrics.baseline_accuracy || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>MACRO-F1</span>
                    <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '4px' }}>
                      {benchmarkData.metrics.debate_macro_f1?.toFixed(3)}
                      <span style={{ fontSize: '13px', color: 'var(--accent-emerald)', marginLeft: '8px' }}>
                        vs {benchmarkData.metrics.baseline_macro_f1?.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>CITATION FAITHFULNESS</span>
                    <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '4px', color: 'var(--accent-cyan)' }}>
                      {((benchmarkData.metrics.avg_citation_validity || 0) * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>BENCHMARK CLAIMS</span>
                    <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '4px' }}>
                      {benchmarkData.metrics.n_claims}
                    </div>
                  </div>
                </div>

                {/* Charts render */}
                {benchmarkData.charts?.length > 0 && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
                    {benchmarkData.charts.map((chartUrl, i) => (
                      <div key={i} style={{ borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                        <img src={chartUrl} alt="Evaluation Chart" style={{ width: '100%', display: 'block' }} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

        </div>
      )}

      {/* Footer */}
      <footer style={{ 
        marginTop: '40px', 
        paddingTop: '20px', 
        borderTop: '1px solid var(--border-color)', 
        display: 'flex', 
        justifyContent: 'space-between',
        fontSize: '12px',
        color: 'var(--text-dim)'
      }}>
        <span>SciFact Dataset & BAAI/bge-base-en-v1.5 + Cross-Encoder Reranker</span>
        <span>Local RTX 3050 CUDA Acceleration Active</span>
      </footer>

    </div>
  );
}
