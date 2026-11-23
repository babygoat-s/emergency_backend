# Cloud Run에 백엔드 배포하기

백엔드 폴더에서 Linux 컨테이너를 빌드하고 서울 리전의 Artifact Registry에 올린 뒤 Cloud Run에 배포한다. 이미지에는 Whisper, 증상·화자 분류 모델과 매뉴얼 인덱스가 들어간다. 빌드에서 `.env`, 샘플 음성, 임시 파일을 제외한다. Cloud Run에서는 음성과 전사를 Google Cloud의 컨테이너에서 처리하며 외부 LLM 호출은 기본으로 꺼져 있다.

## Windows에서 Docker 연결 확인

시작 메뉴에서 **Docker Desktop**을 실행한다. Linux 엔진이 시작되면 PowerShell에서 아래 명령을 실행한다. `Server` 항목이 나오면 Linux 엔진에 연결된 상태다. Windows 컨테이너 모드라면 작업 표시줄의 Docker 메뉴에서 **Switch to Linux containers**를 선택한다. WSL 오류가 표시되면 Docker Desktop의 **Settings → General → Use the WSL 2 based engine**과 WSL 설치 상태를 확인한다. [Docker의 Windows WSL 2 설정](https://docs.docker.com/desktop/features/wsl/)

```powershell
docker --context desktop-linux version
```

## 1. 이미지 빌드 및 배포

Docker Desktop의 Linux 엔진을 켜고 [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install)를 설치한다. 결제가 연결된 프로젝트의 **프로젝트 ID**를 준비한다. 배포 계정에는 Cloud Run 서비스 배포 권한과 실행 서비스 계정 사용 권한, 이미지 저장소 쓰기 권한이 필요하다. API 활성화와 저장소 생성 권한이 없으면 관리자가 두 작업을 먼저 진행하면 된다. [Google Cloud 환경 설정](https://docs.cloud.google.com/run/docs/setup)

```powershell
Set-Location C:\Users\lemon\Desktop\emergency_backend
$projectId = 'YOUR_GOOGLE_CLOUD_PROJECT_ID'
$region = 'asia-northeast3'
$repo = 'emergency-images'
$service = 'emergency-backend'
$registry = "$region-docker.pkg.dev"
$tag = Get-Date -Format 'yyyyMMdd-HHmmss'
$imageRef = "$registry/$projectId/$repo/${service}:$tag"

gcloud auth login
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project $projectId
gcloud artifacts repositories create $repo --repository-format=docker --location $region --project $projectId
gcloud auth configure-docker $registry
```

저장소 생성은 프로젝트당 한 번만 한다. 이어서 같은 터미널에서 빌드하고 배포한다.

```powershell
docker build --platform linux/amd64 --tag $imageRef .
docker push $imageRef

gcloud run deploy $service --image $imageRef --project $projectId --region $region `
  --port 8080 --execution-environment gen2 --no-use-http2 `
  --cpu 4 --memory 8Gi --concurrency 1 --min 0 --max 1 --cpu-throttling `
  --set-env-vars 'REALTIME_ANALYSIS_TIMEOUT_SECONDS=60,REALTIME_ANALYSIS_MIN_INTERVAL_SECONDS=20' `
  --timeout 900s --cpu-boost --invoker-iam-check --no-allow-unauthenticated
```

현재 소스로 빌드한 `emergency-backend:cloud-run-local` 이미지가 이미 있으면 `docker tag emergency-backend:cloud-run-local $imageRef`로 배포용 태그를 붙인 뒤 `docker push`부터 실행할 수 있다.

4 vCPU와 8 GiB는 모델 여러 개를 함께 실행하기 위한 시작값이다. 인스턴스당 요청 하나로 시작해 실제 통화 길이의 테스트에서 전사 지연과 메모리를 확인한다. Cloud Run WebSocket 연결도 요청 제한 시간의 적용을 받아 위 설정에서는 15분 후 종료된다. 녹음 자체는 앱 설정에 따라 최대 10분이다. 파일 업로드 한도는 Cloud Run의 HTTP/1 요청 제한 32 MiB 안에 multipart 헤더도 들어가도록 30 MiB로 설정했다. 첫 요청에서는 새 인스턴스가 시작되면서 모델 로딩 시간이 추가될 수 있다. [WebSocket 문서](https://docs.cloud.google.com/run/docs/triggering/websockets) · [요청 제한](https://docs.cloud.google.com/run/quotas#networking_limits)

## 2. 인증된 요청으로 확인

서비스를 호출할 Google 계정에는 서비스의 `roles/run.invoker`가 필요하다. 로컬에서 Google Cloud CLI 인증을 붙이는 프록시를 아래처럼 실행해 둔다. [비공개 서비스 테스트](https://docs.cloud.google.com/run/docs/authenticating/developers)

```powershell
gcloud run services proxy $service --project $projectId --region $region --port 8080
```

다른 PowerShell 창에서 응답을 확인한다.

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/v1/health
Invoke-RestMethod http://127.0.0.1:8080/api/v1/analysis/config
Invoke-RestMethod http://127.0.0.1:8080/api/v1/rag/status
```

`health`에서 `status: ok`, 설정에서 `llm_external_enabled: false`, 매뉴얼 상태에서 `available: true`를 확인한다. 이어서 테스트 음성으로 파일·실시간 전사를 확인한다.

## 3. 기존 프런트엔드에서 시험

기존 Vite `/api` 프록시에 인증 프록시 주소를 전달한다. 프런트엔드 소스 수정 없이 현재 터미널 세션에서 지정할 수 있다. 기존에 `VITE_API_BASE_URL`을 지정했다면 같은 출처의 `/api`로 요청하도록 해당 값을 비운다.

```powershell
Set-Location C:\Users\lemon\Desktop\emergency_frontend
$env:API_PROXY_TARGET = 'http://127.0.0.1:8080'
npm run dev -- --host 127.0.0.1
```

브라우저에서 `http://127.0.0.1:5173`을 열면 로컬 프런트엔드가 Cloud Run 백엔드로 요청한다. 이 방식은 개발자 테스트에 사용한다. IAM 전용 서비스와 배포된 웹사이트를 연결하는 경우에는 HTTP와 WebSocket에 서버 측 인증을 붙이는 게이트웨이를 사용할 수 있다. 서비스 계정 키는 브라우저 코드에 넣지 않는다. [서비스 간 인증](https://docs.cloud.google.com/run/docs/authenticating/service-to-service)

## 4. Vercel에서 Cloud Run에 직접 연결

1. Google Auth Platform에서 웹 애플리케이션 유형의 OAuth 클라이언트를 만든다. 실제 Vercel HTTPS 주소를 **승인된 자바스크립트 원본**에 등록한다. 로컬에서 먼저 시험할 경우 `http://localhost:5173` 및 `http://127.0.0.1:5173`도 등록한다. 동의 화면이 테스트 상태라면 사용할 Google 계정을 테스트 사용자에 추가한다. [Google 로그인 클라이언트 설정](https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid)
2. 아래 환경변수로 새 백엔드 이미지를 배포하고 Cloud Run의 호출자 IAM 검사를 유지한 상태에서 확인한다. 모든 확인된 Google 계정을 허용하려면 `verified_google_accounts`를 지정한다. `allowlist` 모드에서는 목록에 Gmail이나 Google이 관리하는 Workspace 계정의 이메일을 적는다.

| Cloud Run 환경변수 | 값 예시 |
| --- | --- |
| `GOOGLE_AUTH_REQUIRED` | `true` |
| `GOOGLE_OAUTH_CLIENT_ID` | `123456789-example.apps.googleusercontent.com` |
| `GOOGLE_AUTH_ACCESS` | `verified_google_accounts` 또는 `allowlist` |
| `GOOGLE_ALLOWED_EMAILS` | `allowlist` 모드 전용. 예: `["tester@gmail.com"]` |
| `CORS_ORIGINS` | `["https://your-project.vercel.app","http://localhost:5173","http://127.0.0.1:5173"]` |

3. Vercel 프런트엔드 프로젝트의 `GOOGLE_CLIENT_ID`를 동일하게 지정하고 `BE_API_BASE_URL`에는 Cloud Run 서비스의 HTTPS origin을 넣는다. `VITE_API_BASE_URL`도 설정했다면 그 값이 우선한다. WebSocket과 파일 업로드는 브라우저에서 Cloud Run에 직접 연결한다.
4. 비공개 서비스의 로컬 인증 프록시로 로그인과 파일·WebSocket을 먼저 시험한다. 이때 로컬 프런트엔드는 `VITE_API_BASE_URL`을 비우고 `API_PROXY_TARGET=http://127.0.0.1:8080`을 사용한다. 앱 인증 없는 요청이 거부되는지 확인한 뒤 Cloud Run 콘솔에서 호출자 IAM 검사를 해제하고 Vercel 배포에서도 다시 확인한다. FastAPI는 HTTP와 WebSocket 모두에서 Google 토큰과 접근 설정을 검증한다. [Cloud Run 사용자 인증](https://docs.cloud.google.com/run/docs/authenticating/end-users)