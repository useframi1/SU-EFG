import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { Prompt } from '../_models/prompt';
import { BotResponse } from '../_models/botResponse';

@Injectable({
  providedIn: 'root',
})
export class ChatbotService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiUrl;

  create_session() {
    return this.http.get<string>(this.baseUrl, { withCredentials: true });
  }

  send_prompt(model: Prompt) {
    return this.http.post<BotResponse>(this.baseUrl + 'send_prompt', model, {
      withCredentials: true,
    });
  }
}
