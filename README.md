# ai-inference-server
AI inference server that handles data preprocessing and serves prediction results based on weather and date-related features.

## 프로젝트 설정

### 레포 클론 구조
docker-compose가 ai-server 레포를 상대경로로 참조하므로,
반드시 아래 구조로 클론해야 합니다.
```
workspace/
├── backend-core/   # git clone https://github.com/bid-weather/backend-core.git
└── ai-server/      # git clone https://github.com/bid-weather/ai-server.git
```

### 실행
```bash
cd backend-core
docker compose up -d --build
```