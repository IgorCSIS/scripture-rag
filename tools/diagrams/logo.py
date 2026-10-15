"""Generate site/assets/logo.svg and favicon.svg."""
import math
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "site" / "assets"
cx, cy = 256, 228
rays = []
for i in range(12):
    a = math.radians(i * 30)
    r0, r1 = (64, 150) if i % 2 == 0 else (64, 118)
    x0, y0 = cx + r0 * math.sin(a), cy - r0 * math.cos(a)
    x1, y1 = cx + r1 * math.sin(a), cy - r1 * math.cos(a)
    rays.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>')
logo = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="120" height="120" role="img" aria-labelledby="lt">
<title id="lt">Scripture RAG: an open book with light rising evenly in every direction</title>
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2b3a8c"/><stop offset="1" stop-color="#0b6e79"/></linearGradient>
<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#fff9db"/><stop offset="0.55" stop-color="#ffe066" stop-opacity="0.55"/><stop offset="1" stop-color="#ffe066" stop-opacity="0"/></radialGradient>
</defs>
<circle cx="256" cy="256" r="244" fill="url(#bg)"/>
<circle cx="256" cy="256" r="244" fill="none" stroke="#ffffff" stroke-opacity="0.18" stroke-width="6"/>
<circle cx="{cx}" cy="{cy}" r="120" fill="url(#glow)"/>
<g stroke="#ffd43b" stroke-width="16" stroke-linecap="round">{"".join(rays)}</g>
<circle cx="{cx}" cy="{cy}" r="36" fill="#fff9db"/>
<g fill="#ffffff" stroke="#1d2b6b" stroke-width="6" stroke-linejoin="round">
<path d="M256,426 C212,400 156,392 100,402 L100,318 C156,308 212,316 256,342 Z"/>
<path d="M256,426 C300,400 356,392 412,402 L412,318 C356,308 300,316 256,342 Z"/>
</g>
<g fill="none" stroke="#9aa5c4" stroke-width="5" stroke-linecap="round">
<path d="M130,340 C168,334 204,338 232,352"/><path d="M130,364 C168,358 204,362 232,376"/>
<path d="M382,340 C344,334 308,338 280,352"/><path d="M382,364 C344,358 308,362 280,376"/>
</g>
</svg>
'''
fav = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2b3a8c"/><stop offset="1" stop-color="#0b6e79"/></linearGradient></defs>
<circle cx="16" cy="16" r="15.5" fill="url(#bg)"/>
<circle cx="16" cy="11" r="4.6" fill="#ffd43b"/>
<path d="M16 26.5c-3-1.8-6.6-2.3-10-1.7v-7.6c3.4-.6 7 .1 10 1.9zM16 26.5c3-1.8 6.6-2.3 10-1.7v-7.6c-3.4-.6-7 .1-10 1.9z" fill="#ffffff"/>
</svg>
'''
(R / "logo.svg").write_text(logo)
(R / "favicon.svg").write_text(fav)
