FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM node:20-alpine
RUN npm install -g serve
COPY --from=builder /app/dist /app/dist
WORKDIR /app
CMD ["sh", "-c", "serve -s dist -l ${PORT:-3000}"]
