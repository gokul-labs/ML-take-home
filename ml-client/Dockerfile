FROM node:13.12.0-alpine

WORKDIR /ml-client

COPY package.json ./
COPY package-lock.json ./
COPY postcss.config.js ./
COPY tailwind.config.js ./
COPY tsconfig.json ./
COPY public/ ./public/
COPY src/ ./src/
RUN npm install --silent

EXPOSE 3000

CMD ["npm", "start"]
