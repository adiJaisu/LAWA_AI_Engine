FROM node:25-slim

RUN npm install -g npm@11.6.4

# ensure directory exists before changing ownership
RUN mkdir -p /frontend/ai_vms
RUN chown -R node:node /frontend/ai_vms

WORKDIR /frontend/ai_vms

USER root
COPY ai_vms/package*.json ./
RUN chown -R node:node /frontend/ai_vms

USER node
RUN npm install --include dev

COPY --chown=node:node ai_vms/ .

CMD ["npm", "run", "dev"]